import sys
import os
from pathlib import Path
import traceback

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.morphological import run_morphological_pipeline
from src.directional_gradient import run_directional_gradient_pipeline
from src.feature_extraction import BlobFeatureExtractor, LineFeatureExtractor

app = FastAPI(title="Fabric Defect Detection ML Service")

# Load models at startup
MODELS_DIR = PROJECT_ROOT / "models"
try:
    rf_morph = joblib.load(MODELS_DIR / "rf_morphological_model.pkl")
    rf_directional = joblib.load(MODELS_DIR / "rf_directional_model.pkl")
    svm_morph = joblib.load(MODELS_DIR / "svm_morphological_model.pkl")
    svm_directional = joblib.load(MODELS_DIR / "svm_directional_model.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load models. Error: {e}")
    rf_morph = None
    rf_directional = None
    svm_morph = None
    svm_directional = None

class PredictRequest(BaseModel):
    image_path: str
    pipeline_type: str
    output_dir: str
    model_type: str = "rf"

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        img_path = Path(request.image_path)
        out_dir = Path(request.output_dir)
        pipeline_type = request.pipeline_type.lower()
        model_type = request.model_type.lower()
        
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
            
        out_dir.mkdir(parents=True, exist_ok=True)
        
        defect_type = "Unknown"
        confidence = 0.0
        features_dict = {}
        images_result = {}
        
        # Base filenames for the saved overlay and mask
        # Example: if original is "test.png", overlay will be "test_overlay.png"
        basename = img_path.stem
        overlay_filename = f"{basename}_overlay.png"
        overlay_path = out_dir / overlay_filename
        
        if pipeline_type == "morphological":
            selected_model = rf_morph if model_type == "rf" else svm_morph
            if selected_model is None:
                raise HTTPException(status_code=500, detail="Morphological model not loaded")
                
            # Run pipeline
            results = run_morphological_pipeline(
                img_path, 
                output_dir=None,
                median_kernel=7, 
                tophat_kernel=21, 
                k_std=3.5, 
                open_size=3, 
                close_size=5, 
                iterations=3, 
                resize_width=512
            ) # We only need memory output
            
            original_gray = results["gray"]
            binary_mask = results["morph_closing"]
            
            # Extract features
            extractor = BlobFeatureExtractor(original_gray, binary_mask)
            extractor.extract_contours()
            features_dict = extractor.compute_features(img_path.name, "unknown")
            
            # Save overlay
            extractor.save_overlay(overlay_path)
            
            # Predict
            # Order of features must match training. From feature_extraction.py:
            # max_area, total_area, max_perimeter, min_eccentricity
            if not features_dict or features_dict.get('total_area', 0) == 0:
                defect_type = "Good"
                confidence = 1.0
            else:
                feature_vector = np.array([[
                    features_dict["max_area"],
                    features_dict["total_area"],
                    features_dict["max_perimeter"],
                    features_dict["min_eccentricity"]
                ]])
                
                prediction = selected_model.predict(feature_vector)[0]
                proba = selected_model.predict_proba(feature_vector)[0]
                defect_type = str(prediction)
                confidence = float(np.max(proba))
                
            images_result = {
                "overlay": overlay_filename
            }
            
        elif pipeline_type == "directional":
            selected_model = rf_directional if model_type == "rf" else svm_directional
            if selected_model is None:
                raise HTTPException(status_code=500, detail="Directional model not loaded")
                
            results = run_directional_gradient_pipeline(
                img_path, 
                output_dir=None,
                median_ksize=7, 
                apply_tophat=True, 
                tophat_ksize=21, 
                resize_width=512
            )
            
            original_gray = results["gray"]
            horiz_mask = results["horiz_mask"]
            vert_mask = results["vert_mask"]
            diag_mask = results["diag_mask"]
            
            extractor = LineFeatureExtractor(original_gray, horiz_mask, vert_mask, diag_mask)
            features_dict = extractor.compute_features(img_path.name, "unknown")
            
            extractor.save_overlay(overlay_path)
            
            # Predict
            # Features: horiz_length, vert_length, diag_length
            if not features_dict or (features_dict.get('horiz_length', 0) == 0 and 
                                     features_dict.get('vert_length', 0) == 0 and 
                                     features_dict.get('diag_length', 0) == 0):
                defect_type = "Good"
                confidence = 1.0
            else:
                feature_vector = np.array([[
                    features_dict["horiz_length"],
                    features_dict["vert_length"],
                    features_dict["diag_length"]
                ]])
                
                prediction = selected_model.predict(feature_vector)[0]
                proba = selected_model.predict_proba(feature_vector)[0]
                defect_type = str(prediction)
                confidence = float(np.max(proba))
                
            images_result = {
                "overlay": overlay_filename
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid pipeline_type. Must be 'morphological' or 'directional'")

        return {
            "defect_type": defect_type,
            "confidence": confidence,
            "features": features_dict,
            "images": images_result
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
