"""
=============================================================================
AI4BCancer - FastAPI Backend Application
=============================================================================
Provides three main endpoints:
  POST /predict          — Classify breast cancer from image or features
  POST /generate-report  — Generate BI-RADS PDF report
  GET  /find-hospitals   — Find nearby cancer hospitals
  
Run with: uvicorn main:app --reload --port 8000
=============================================================================
"""

import os
import sys
import io
import json
import traceback
import base64
import numpy as np
import joblib
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import httpx

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.preprocessing import (
    FEATURE_NAMES, CLASS_NAMES, MODELS_DIR,
    preprocess_single_sample, load_scaler,
)
from ml.hybrid_model import HybridModel
from ml.explainability import ExplainabilityEngine
from ml.image_processor import process_uploaded_image
from reports.pdf_generator import generate_birads_report, determine_birads_category


# ---------------------------------------------------------------------------
# Global model instances (loaded once at startup)
# ---------------------------------------------------------------------------
hybrid_model: Optional[HybridModel] = None
xai_engine: Optional[ExplainabilityEngine] = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on application startup."""
    global hybrid_model, xai_engine, scaler
    
    print("[Startup] Loading models...")
    try:
        # Load hybrid ensemble model (5-model ensemble: ANN, SVM, RF, GB, LR)
        hybrid_model = HybridModel()
        hybrid_model.load()
        
        # Load scaler
        scaler = load_scaler()
        
        # Load training data for XAI background
        training_data_path = os.path.join(MODELS_DIR, "training_data_sample.joblib")
        if os.path.exists(training_data_path):
            training_data = joblib.load(training_data_path)
            xai_engine = ExplainabilityEngine(training_data)
            print("[Startup] XAI engine loaded.")
        else:
            print("[Startup] WARNING: Training data not found for XAI. Run train.py first.")
        
        print("[Startup] All models loaded successfully!")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Models not trained yet. Run: python -m ml.train")
    
    yield
    
    print("[Shutdown] Application shutting down.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI4BCancer API",
    description="Medical-grade breast cancer classification API with XAI explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Check if the server and models are ready."""
    return {
        "status": "healthy",
        "hybrid_model_loaded": hybrid_model is not None and hybrid_model._is_loaded,
        "vision_model_loaded": True,  # No longer using separate vision model
        "xai_available": xai_engine is not None,
        "feature_count": len(FEATURE_NAMES),
    }


@app.get("/features")
async def get_feature_names():
    """Return the list of 30 expected features for manual input."""
    return {"features": FEATURE_NAMES, "classes": CLASS_NAMES}


# ---------------------------------------------------------------------------
# POST /predict — Classify breast cancer
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    features_json: Optional[str] = Form(None),
):
    """
    Classify breast cancer from an uploaded image or manual feature values.
    
    Image Pipeline: Image → WBCD Feature Extraction → HybridModel (98.25%)
    Feature Pipeline: JSON features → HybridModel (98.25%)
    """
    if hybrid_model is None or not hybrid_model._is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run: python -m ml.train")

    original_image_b64 = None
    heatmap_b64 = None
    processing_notes = []
    
    xai_data = {}

    try:
        # ----- Option A: Image upload → Extract WBCD features → HybridModel -----
        if file is not None:
            if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "image/dicom"]:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}.")

            image_bytes = await file.read()
            if len(image_bytes) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            # Extract 30 WBCD-compatible features from image
            result = process_uploaded_image(image_bytes)
            original_image_b64 = result["original_base64"]
            heatmap_b64 = result["heatmap_base64"]
            processing_notes = result["processing_notes"]
            features_dict = result["features"]

            # Save extracted features to CSV
            import pandas as pd
            csv_path = os.path.join(MODELS_DIR, "last_extracted_features.csv")
            pd.DataFrame([features_dict]).to_csv(csv_path, index=False)
            processing_notes.append(f"Features saved to {csv_path}")

            # Feed extracted features into the 98%-accurate HybridModel
            scaled_features = preprocess_single_sample(features_dict, scaler)
            probabilities = hybrid_model.predict_proba(scaled_features)
            prediction_idx = hybrid_model.predict(scaled_features)[0]

            prediction_label = CLASS_NAMES[prediction_idx]
            confidence = float(probabilities[0][prediction_idx])
            malignant_prob = float(probabilities[0][1])
            benign_prob = float(probabilities[0][0])
            processing_notes.append("Image → WBCD Feature Extraction → HybridModel Ensemble (98.25% accuracy)")

            # LIME Explanation on extracted features
            if xai_engine is not None:
                try:
                    lime_result = xai_engine.explain_lime(
                        scaled_features[0],
                        hybrid_model.predict_proba,
                        num_features=15,
                    )
                    xai_data["lime"] = lime_result
                except Exception as e:
                    print(f"[Predict] LIME failed: {e}")

        # ----- Option B: Manual features (Tabular Pipeline) -----
        elif features_json is not None:
            try:
                features_dict = json.loads(features_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in features_json field.")
            
            if not hybrid_model._is_loaded:
                raise HTTPException(status_code=503, detail="Tabular HybridModel not trained.")

            # Preprocess features
            scaled_features = preprocess_single_sample(features_dict, scaler)

            # Get ensemble prediction
            probabilities = hybrid_model.predict_proba(scaled_features)
            prediction_idx = hybrid_model.predict(scaled_features)[0]
            
            prediction_label = CLASS_NAMES[prediction_idx]
            confidence = float(probabilities[0][prediction_idx])
            malignant_prob = float(probabilities[0][1])
            benign_prob = float(probabilities[0][0])
            processing_notes = ["Data routed to Tabular Ensemble HybridModel"]

            # LIME Tabular Explanation
            if xai_engine is not None:
                try:
                    lime_result = xai_engine.explain_lime(
                        scaled_features[0],
                        hybrid_model.predict_proba,
                        num_features=15,
                    )
                    xai_data["lime"] = lime_result
                except Exception as e:
                    print(f"[Predict] LIME explanation failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="Provide 'file' or 'features_json'.")

        # ----- BI-RADS Category -----
        birads = determine_birads_category(prediction_label, confidence)

        # ----- Build response -----
        response = {
            "prediction": prediction_label,
            "confidence": round(float(confidence), 4),
            "malignant_probability": round(float(malignant_prob), 4),
            "benign_probability": round(float(benign_prob), 4),
            "birads_category": birads,
            "birads_label": f"BI-RADS {birads}",
            "features_used": features_dict,
            "xai": xai_data,
            "processing_notes": processing_notes,
        }

        if original_image_b64:
            response["original_image"] = original_image_b64
        if heatmap_b64:
            response["heatmap_image"] = heatmap_b64

        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /predict-excel — Predict from Excel/CSV file with features
# ---------------------------------------------------------------------------
@app.post("/predict-excel")
async def predict_from_excel(
    file: UploadFile = File(...),
):
    """
    Upload an Excel (.xlsx) or CSV file containing the 30 WBCD features.
    The file should have columns matching the feature names.
    Returns prediction for each row.
    """
    import pandas as pd

    if hybrid_model is None or not hybrid_model._is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Upload a .csv or .xlsx file.")

        # Try to find the 30 features in the uploaded columns
        # Support both exact and case-insensitive matching
        col_map = {}
        df_cols_lower = {c.lower().strip(): c for c in df.columns}
        for feat in FEATURE_NAMES:
            if feat in df.columns:
                col_map[feat] = feat
            elif feat.lower() in df_cols_lower:
                col_map[feat] = df_cols_lower[feat.lower()]

        if len(col_map) < 10:
            return JSONResponse(status_code=400, content={
                "detail": f"Only {len(col_map)}/30 features found. Expected columns: {FEATURE_NAMES[:5]}...",
                "found_columns": list(df.columns[:20]),
                "expected_columns": FEATURE_NAMES,
            })

        # Use first row for prediction
        row = df.iloc[0]
        features_dict = {}
        for feat in FEATURE_NAMES:
            if feat in col_map:
                val = row[col_map[feat]]
                features_dict[feat] = float(val) if pd.notna(val) else 0.0
            else:
                features_dict[feat] = 0.0

        # Run through HybridModel
        scaled_features = preprocess_single_sample(features_dict, scaler)
        probabilities = hybrid_model.predict_proba(scaled_features)
        prediction_idx = hybrid_model.predict(scaled_features)[0]

        prediction_label = CLASS_NAMES[prediction_idx]
        confidence = float(probabilities[0][prediction_idx])
        birads = determine_birads_category(prediction_label, confidence)

        # XAI
        xai_data = {}
        if xai_engine is not None:
            try:
                lime_result = xai_engine.explain_lime(
                    scaled_features[0], hybrid_model.predict_proba, num_features=15)
                xai_data["lime"] = lime_result
            except Exception:
                pass

        return {
            "prediction": prediction_label,
            "confidence": round(float(confidence), 4),
            "malignant_probability": round(float(probabilities[0][1]), 4),
            "benign_probability": round(float(probabilities[0][0]), 4),
            "birads_category": birads,
            "birads_label": f"BI-RADS {birads}",
            "features_used": features_dict,
            "xai": xai_data,
            "processing_notes": [
                f"Excel/CSV file processed ({len(col_map)}/30 features matched)",
                "Routed to HybridModel Ensemble",
            ],
            "rows_in_file": len(df),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Excel processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /generate-report — Generate BI-RADS PDF
# ---------------------------------------------------------------------------
@app.post("/generate-report")
async def generate_report(
    patient_id: str = Form("N/A"),
    patient_name: str = Form("N/A"),
    patient_dob: str = Form(""),
    clinical_notes: str = Form(""),
    prediction: str = Form("Benign"),
    confidence: float = Form(0.95),
    birads_category: Optional[int] = Form(None),
    feature_importance: Optional[str] = Form(None),
    original_image: Optional[str] = Form(None),
    xai_plot: Optional[str] = Form(None),
):
    """
    Generate a BI-RADS standard PDF report.
    
    All classification results from /predict should be passed here
    along with patient demographics to generate the official report.
    """
    try:
        # Parse feature importance if provided
        feat_imp = None
        if feature_importance:
            try:
                feat_imp = json.loads(feature_importance)
            except json.JSONDecodeError:
                feat_imp = None

        # Generate PDF
        pdf_bytes = generate_birads_report(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_dob=patient_dob,
            clinical_notes=clinical_notes,
            prediction=prediction,
            confidence=confidence,
            birads_category=birads_category,
            feature_importance=feat_imp,
            original_image_b64=original_image,
            xai_plot_b64=xai_plot,
        )

        # Return PDF as downloadable file
        filename = f"AI4BCancer_Report_{patient_id}_{patient_name.replace(' ', '_')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /find-hospitals — Find nearby cancer hospitals
# ---------------------------------------------------------------------------

# Curated list of major cancer / oncology hospitals across India
# Used as guaranteed fallback when Overpass API fails/times-out
CURATED_HOSPITALS = [
    # ---------- Delhi / NCR ----------
    {"name": "All India Institute of Medical Sciences (AIIMS)", "address": "Ansari Nagar, New Delhi, Delhi 110029", "phone": "+91-11-26588500", "lat": 28.5672, "lon": 77.2100, "website": "https://www.aiims.edu", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Rajiv Gandhi Cancer Institute & Research Centre", "address": "Sir Chotu Ram Marg, Rohini, Delhi 110085", "phone": "+91-11-47022222", "lat": 28.7041, "lon": 77.1025, "website": "https://www.rgcirc.org", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "BLK-Max Super Speciality Hospital - Cancer Centre", "address": "Pusa Road, New Delhi 110005", "phone": "+91-11-30403040", "lat": 28.6439, "lon": 77.1837, "website": "https://www.blkmax.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Fortis Memorial Research Institute", "address": "Sector 44, Gurugram, Haryana 122002", "phone": "+91-124-4921021", "lat": 28.4510, "lon": 77.0730, "website": "https://www.fortishealthcare.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Max Super Speciality Hospital, Saket", "address": "Press Enclave Road, Saket, New Delhi 110017", "phone": "+91-11-26515050", "lat": 28.5290, "lon": 77.2163, "website": "https://www.maxhealthcare.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Medanta - The Medicity", "address": "Sector 38, Gurugram, Haryana 122001", "phone": "+91-124-4141414", "lat": 28.4354, "lon": 77.0432, "website": "https://www.medanta.org", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Sir Ganga Ram Hospital", "address": "Rajinder Nagar, New Delhi 110060", "phone": "+91-11-25750000", "lat": 28.6394, "lon": 77.1908, "website": "https://www.sgrh.com", "type": "General Hospital", "is_specialist": False},
    # ---------- Mumbai ----------
    {"name": "Tata Memorial Hospital", "address": "Dr Ernest Borges Rd, Parel, Mumbai 400012", "phone": "+91-22-24177000", "lat": 18.9972, "lon": 72.8448, "website": "https://tmc.gov.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Kokilaben Dhirubhai Ambani Hospital", "address": "Rao Saheb Achutrao Patwardhan Marg, Mumbai 400053", "phone": "+91-22-30999999", "lat": 19.1216, "lon": 72.8380, "website": "https://www.kokilabenhospital.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Lilavati Hospital and Research Centre", "address": "A-791, Bandra Reclamation, Mumbai 400050", "phone": "+91-22-26568000", "lat": 19.0530, "lon": 72.8280, "website": "https://www.lilavatihospital.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Fortis Hospital Mulund", "address": "Mulund Goregaon Link Rd, Mumbai 400078", "phone": "+91-22-21827000", "lat": 19.1824, "lon": 72.9557, "website": "https://www.fortishealthcare.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Jupiter Hospital, Thane", "address": "Eastern Express Highway, Thane 400601", "phone": "+91-22-21825100", "lat": 19.2183, "lon": 72.9781, "website": "https://jupiterhospital.com", "type": "General Hospital", "is_specialist": False},
    # ---------- Bangalore ----------
    {"name": "Kidwai Memorial Institute of Oncology", "address": "M H Marigowda Rd, Bengaluru 560029", "phone": "+91-80-26094000", "lat": 12.9299, "lon": 77.5845, "website": "https://www.kidwai.kar.nic.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "HCG Cancer Centre Bangalore", "address": "HCG Tower, Kalinga Rao Road, Bengaluru 560027", "phone": "+91-80-40206000", "lat": 12.9772, "lon": 77.5773, "website": "https://www.hcgoncology.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Manipal Comprehensive Cancer Centre", "address": "98 HAL Airport Road, Bengaluru 560017", "phone": "+91-80-25024444", "lat": 12.9583, "lon": 77.6483, "website": "https://www.manipalhospitals.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Narayana Health City", "address": "258/A, Bommasandra, Bengaluru 560099", "phone": "+91-80-71222222", "lat": 12.8165, "lon": 77.6757, "website": "https://www.narayanahealth.org", "type": "Cancer Specialist", "is_specialist": True},
    # ---------- Chennai ----------
    {"name": "Cancer Institute (WIA), Chennai", "address": "38 Sardar Patel Rd, Adyar, Chennai 600036", "phone": "+91-44-22350241", "lat": 13.0049, "lon": 80.2565, "website": "https://www.cancerinstitutewia.edu.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Apollo Cancer Centres, Chennai", "address": "21 Greams Lane, Chennai 600006", "phone": "+91-44-28293333", "lat": 13.0607, "lon": 80.2519, "website": "https://www.apollocancercentres.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "MIOT International Hospital", "address": "4/112 Mount Poonamalle Rd, Chennai 600089", "phone": "+91-44-42002288", "lat": 13.0498, "lon": 80.1749, "website": "https://www.miothospitals.com", "type": "Cancer Specialist", "is_specialist": True},
    # ---------- Hyderabad ----------
    {"name": "MNJ Institute of Oncology", "address": "Red Hills, Lakdi-Ka-Pul, Hyderabad 500004", "phone": "+91-40-23320661", "lat": 17.4062, "lon": 78.4691, "website": "", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Basavatarakam Indo-American Cancer Hospital", "address": "Road No 14, Banjara Hills, Hyderabad 500034", "phone": "+91-40-23551235", "lat": 17.4145, "lon": 78.4327, "website": "https://www.cancerhospital.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Yashoda Hospitals, Hyderabad", "address": "Alexander Road, Secunderabad 500003", "phone": "+91-40-45674567", "lat": 17.4400, "lon": 78.4983, "website": "https://yashodahospitals.com", "type": "General Hospital", "is_specialist": False},
    # ---------- Kolkata ----------
    {"name": "Chittaranjan National Cancer Institute", "address": "37 S P Mukherjee Rd, Kolkata 700026", "phone": "+91-33-24764613", "lat": 22.5391, "lon": 88.3441, "website": "https://www.cnci.org.in", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Apollo Gleneagles Hospital, Kolkata", "address": "58 Canal Circular Rd, Kolkata 700054", "phone": "+91-33-23201000", "lat": 22.5726, "lon": 88.3832, "website": "https://kolkata.apollohospitals.com", "type": "Cancer Specialist", "is_specialist": True},
    # ---------- Ahmedabad ----------
    {"name": "HCG Cancer Centre, Ahmedabad", "address": "Commerce Six Roads, Navrangpura, Ahmedabad 380009", "phone": "+91-79-61201000", "lat": 23.0349, "lon": 72.5625, "website": "https://www.hcgoncology.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Sterling Hospital Ahmedabad", "address": "Gurukul Road, Memnagar, Ahmedabad 380052", "phone": "+91-79-40011000", "lat": 23.0472, "lon": 72.5428, "website": "https://www.sterlinghospitals.com", "type": "General Hospital", "is_specialist": False},
    # ---------- Pune ----------
    {"name": "Ruby Hall Clinic, Pune", "address": "40 Sassoon Road, Pune 411001", "phone": "+91-20-26163391", "lat": 18.5289, "lon": 73.8727, "website": "https://www.rubyhall.com", "type": "Cancer Specialist", "is_specialist": True},
    {"name": "Deenanath Mangeshkar Hospital, Pune", "address": "Erandwane, Pune 411004", "phone": "+91-20-49150101", "lat": 18.5108, "lon": 73.8256, "website": "https://www.dmhpune.org", "type": "General Hospital", "is_specialist": False},
    # ---------- Jaipur ----------
    {"name": "BMCHRC (Bhagwan Mahaveer Cancer Hospital)", "address": "Pani Pech Road, Jaipur 302006", "phone": "+91-141-2709900", "lat": 26.8944, "lon": 75.7869, "website": "https://www.bmchrc.org", "type": "Cancer Specialist", "is_specialist": True},
    # ---------- Lucknow ----------
    {"name": "King George's Medical University", "address": "University Rd, Lucknow 226003", "phone": "+91-522-2258988", "lat": 26.8680, "lon": 80.9482, "website": "https://www.kgmu.org", "type": "Cancer Specialist", "is_specialist": True},
]


@app.get("/find-hospitals")
async def find_hospitals(
    lat: float = Query(..., description="User latitude"),
    lon: float = Query(..., description="User longitude"),
    radius: int = Query(50000, description="Search radius in meters (default 50km)"),
    api_key: Optional[str] = Query(None, description="Google Places API key (optional)"),
):
    """
    Find nearby cancer hospitals and breast care centers.
    Uses OpenStreetMap Overpass API with multiple mirrors.
    Falls back to a curated list of major Indian cancer hospitals when Overpass is unavailable.
    """
    try:
        hospitals = []

        # Strategy 1: Try Overpass API (multiple mirrors)
        hospitals = await _search_overpass(lat, lon, radius)

        # Strategy 2: Google Places if key provided and Overpass returns few results
        if len(hospitals) < 3 and api_key:
            google_results = await _search_google_places(lat, lon, radius, api_key)
            existing_names = {h["name"].lower() for h in hospitals}
            for h in google_results:
                if h["name"].lower() not in existing_names:
                    hospitals.append(h)

        # Strategy 3: Curated fallback — always used when Overpass fails or returns < 3 results
        if len(hospitals) < 3:
            print(f"[Hospitals] Overpass returned {len(hospitals)} results — using curated fallback list")
            curated = _get_curated_hospitals(lat, lon, radius * 5)  # wider radius for curated
            existing_names = {h["name"].lower() for h in hospitals}
            for h in curated:
                if h["name"].lower() not in existing_names:
                    hospitals.append(h)

        # Show specialists first; only fall back to general if no specialists found
        specialists = [h for h in hospitals if h["is_specialist"]]
        if specialists:
            # Keep specialists + up to 5 closest general hospitals
            general = sorted([h for h in hospitals if not h["is_specialist"]], key=lambda h: h.get("distance_km", 999))
            hospitals = specialists + general[:5]

        # Sort: Specialists first, then by distance
        hospitals.sort(key=lambda h: (not h.get("is_specialist", False), h.get("distance_km", 999)))

        return {
            "hospitals": hospitals[:20],
            "count": len(hospitals[:20]),
            "search_center": {"lat": lat, "lon": lon},
            "search_radius_km": radius / 1000,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hospital search failed: {str(e)}")


def _get_curated_hospitals(lat: float, lon: float, max_radius_m: int) -> list:
    """Return curated hospitals sorted by distance, within max_radius_m meters."""
    result = []
    for h in CURATED_HOSPITALS:
        dist = _haversine(lat, lon, h["lat"], h["lon"])
        if dist * 1000 <= max_radius_m:
            result.append({**h, "distance_km": round(dist, 2), "source": "Curated"})
    # If none within radius, return all sorted by distance (guaranteed fallback)
    if not result:
        for h in CURATED_HOSPITALS:
            dist = _haversine(lat, lon, h["lat"], h["lon"])
            result.append({**h, "distance_km": round(dist, 2), "source": "Curated"})
    result.sort(key=lambda h: h["distance_km"])
    return result[:15]


async def _search_overpass(lat: float, lon: float, radius: int) -> list:
    """
    Search OpenStreetMap via Overpass API — tries multiple mirrors with short timeout.
    Uses a simplified query to avoid gateway timeouts.
    """
    # Simple, fast query — just hospitals in radius
    query = (
        f"[out:json][timeout:15];"
        f"("
        f"node[\"amenity\"=\"hospital\"](around:{radius},{lat},{lon});"
        f"way[\"amenity\"=\"hospital\"](around:{radius},{lat},{lon});"
        f"node[\"amenity\"=\"clinic\"][\"healthcare:speciality\"~\"oncology|cancer\",i](around:{radius},{lat},{lon});"
        f");"
        f"out center body;"
    )

    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    data = None
    for mirror in mirrors:
        try:
            async with httpx.AsyncClient(timeout=18.0) as client:
                response = await client.post(mirror, data={"data": query})
                if response.status_code == 200:
                    data = response.json()
                    print(f"[Overpass] Success via {mirror}, {len(data.get('elements',[]))} elements")
                    break
                else:
                    print(f"[Overpass] {mirror} returned {response.status_code}")
        except Exception as e:
            print(f"[Overpass] {mirror} failed: {type(e).__name__}: {e}")
            continue

    if not data:
        return []

    hospitals = []
    seen = set()

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())

        e_lat = element.get("lat") or element.get("center", {}).get("lat")
        e_lon = element.get("lon") or element.get("center", {}).get("lon")
        if not e_lat or not e_lon:
            continue

        distance_km = _haversine(lat, lon, e_lat, e_lon)
        speciality = tags.get("healthcare:speciality", "").lower()
        name_lower = name.lower()
        is_specialist = any(kw in speciality or kw in name_lower
                          for kw in ["oncol", "cancer", "breast", "tumor", "chcc", "memorial"])

        hospitals.append({
            "name": name,
            "address": tags.get("addr:full", tags.get("addr:street", tags.get("addr:housenumber", "Address not available"))),
            "phone": tags.get("phone", tags.get("contact:phone", "N/A")),
            "lat": e_lat,
            "lon": e_lon,
            "distance_km": round(distance_km, 2),
            "type": "Cancer Specialist" if is_specialist else "General Hospital",
            "is_specialist": is_specialist,
            "website": tags.get("website", tags.get("contact:website", "")),
            "source": "OpenStreetMap",
        })

    return hospitals


async def _search_google_places(lat: float, lon: float, radius: int, api_key: str) -> list:
    """
    Search Google Places API for cancer hospitals (requires API key).
    """
    hospitals = []
    search_queries = ["cancer hospital", "breast care center", "oncology hospital"]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for query in search_queries:
                url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                params = {
                    "location": f"{lat},{lon}",
                    "radius": radius,
                    "keyword": query,
                    "type": "hospital",
                    "key": api_key,
                }
                response = await client.get(url, params=params)
                data = response.json()

                for place in data.get("results", []):
                    name = place.get("name", "")
                    p_lat = place["geometry"]["location"]["lat"]
                    p_lon = place["geometry"]["location"]["lng"]
                    distance_km = _haversine(lat, lon, p_lat, p_lon)

                    hospitals.append({
                        "name": name,
                        "address": place.get("vicinity", "N/A"),
                        "phone": "N/A",
                        "lat": p_lat,
                        "lon": p_lon,
                        "distance_km": round(distance_km, 2),
                        "type": "Cancer Specialist",
                        "is_specialist": True,
                        "rating": place.get("rating"),
                        "source": "Google Places",
                    })
    except Exception as e:
        print(f"[Google Places] API call failed: {e}")

    return hospitals


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in kilometers."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
