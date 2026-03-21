"""
Model Predictor - Handles model loading and inference
Uses FastAI for optimal performance
"""

import json
import io
from pathlib import Path
import logging
from PIL import Image
from PIL import UnidentifiedImageError
import torch
from fastai.vision.all import *

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Loads and uses the trained ConvNeXt model for predictions"""

    def __init__(self):
        """Initialize the predictor and load model"""

        self.model_dir = Path(__file__).parent.parent / "model" / "models"

        # Map between class indices and readable names
        self.class_mapping = {
            "Cassava___bacterial_blight": "Bacterial Blight",
            "Cassava___brown_streak_disease": "Brown Streak Disease",
            "Cassava___green_mottle": "Green Mottle",
            "Cassava___healthy": "Healthy",
            "Cassava___mosaic_disease": "Mosaic Disease"
        }

        self.classes = list(self.class_mapping.keys())

        # Disease information for recommendations
        self.disease_info = {
            "Cassava___bacterial_blight": {
                "severity": "High",
                "description": "Angular leaf spots, yellow halos",
                "action": "Use disease-free cuttings, copper bactericides"
            },
            "Cassava___brown_streak_disease": {
                "severity": "Critical",
                "description": "Yellow streaks, necrotic root lesions",
                "action": "Destroy infected plants immediately"
            },
            "Cassava___green_mottle": {
                "severity": "Moderate",
                "description": "Mosaic and mottling on leaves",
                "action": "Control whitefly vectors, use resistant varieties"
            },
            "Cassava___healthy": {
                "severity": "None",
                "description": "No visible disease signs",
                "action": "Continue monitoring and good practices"
            },
            "Cassava___mosaic_disease": {
                "severity": "High",
                "description": "Mosaic patterns and distortion",
                "action": "Use resistant varieties, control whiteflies"
            }
        }

        try:
            self.learn = self._load_model()
            self.classes = list(self.learn.dls.vocab)
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def _load_model(self):
        """Load FastAI learner from exported model"""

        model_path = self.model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Please run: python model/train.py"
            )

        learn = load_learner(model_path)
        return learn

    def predict(self, image_stream):
        """
        Make prediction on image

        Args:
            image_stream: BytesIO object or PIL Image

        Returns:
            dict: Prediction results with disease, confidence, and probabilities
        """

        try:
            if isinstance(image_stream, (bytes, bytearray)):
                stream = io.BytesIO(image_stream)
            else:
                stream = image_stream
            try:
                stream.seek(0)
            except Exception:
                pass
            try:
                image = Image.open(stream)
                image.load()
            except UnidentifiedImageError as e:
                logger.error("Invalid image data")
                raise ValueError("Unsupported or corrupted image. Please upload a valid JPG or PNG.") from e
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img = PILImage.create(image)
            pred_class, pred_idx, probs = self.learn.predict(img)
            disease_folder = pred_class
            disease_name = self.class_mapping.get(
                disease_folder,
                disease_folder.replace("Cassava___", "").replace("_", " ")
            )
            confidence = float(probs[pred_idx])
            all_probs = {}
            for i, cls in enumerate(self.classes):
                readable_name = self.class_mapping.get(
                    cls,
                    cls.replace("Cassava___", "").replace("_", " ")
                )
                all_probs[readable_name] = float(probs[i])
            disease_details = self.disease_info.get(disease_folder, {})
            result = {
                "disease": disease_name,
                "disease_folder": disease_folder,
                "confidence": confidence,
                "confidence_pct": f"{confidence*100:.2f}%",
                "all_predictions": all_probs,
                "severity": disease_details.get("severity", "Unknown"),
                "description": disease_details.get("description", ""),
                "action": disease_details.get("action", "")
            }
            logger.info(f"Prediction: {disease_name} ({confidence:.2%})")
            return result
        except Exception as e:
            logger.exception("Prediction error")
            raise ValueError(f"Failed to process image: {type(e).__name__}: {e}")

    def predict_batch(self, images):
        """
        Make predictions on multiple images

        Args:
            images: List of image streams

        Returns:
            list: List of prediction results
        """
        results = []
        for img in images:
            try:
                result = self.predict(img)
                results.append({"success": True, "data": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})

        return results
