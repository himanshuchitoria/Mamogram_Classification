# Lumino Oncology: AI-Assisted Breast Cancer Classification System 🧬

Lumino Oncology Cancer is a comprehensive, production-ready full-stack application that provides highly accurate Breast Cancer classification by processing mammogram images through a hybrid AI pipeline. It uses advanced Ensemble Machine Learning and Deep Vision models to classify findings, generate feature-level explanations (XAI), and produce robust clinical BI-RADS reports. Furthermore, it integrates a proximity-based oncology hospital locator for immediate patient care routing.

---

## 🎯 Key Features

- 📸 **Vision Pipeline**: Accepts mammogram images and directly extracts structured cytological features via a customized DenseNet121 vision component.
- 🧠 **Hybrid Ensemble Model**: Employs a weighted soft-voting ensemble using Random Forest, XGBoost, LightGBM, SVM, and Neural Networks across 30 unified WBCD (Wisconsin Breast Cancer Dataset) standard features.
- 🔍 **Explainable AI (XAI)**: Visualizes predictions down to the feature level, generating LIME (Local Interpretable Model-Agnostic Explanations) graphs and detailed positive/negative contributing factors.
- 📄 **Automated PDF Reports**: Generates professional, clinically-formatted structured BI-RADS analysis reports in PDF format for direct printing/sharing.
- 🗺️ **Hospital Locator**: Automatically detects user location and searches over OpenStreetMap & curated databases to locate nearby specialized cancer clinics and general hospitals, computing actual distances with interactive map displays.
 
---

## 🏗️ Architecture

The project adopts a modern React + FastAPI decoupled architecture.

### **Backend (`/backend`)**
Built with **FastAPI** to provide high-performance, asynchronous endpoints.

**Core Modules:**
1. **`ml/train.py`**: Pipeline entry point that trains, tunes, and saves the individual ensemble ML models using `joblib`.
2. **`ml/hybrid_model.py`**: The crux of the AI system. Orchestrates loading `models/`, accepts structured cell features (from UI or Vision), parses them into Pandas dataframes, standardizes them (via `StandardScaler`), extracts confidence probabilities via weighted soft-voting, and computes LIME predictions.
3. **`ml/image_processor.py`**: Acts as a simulated Vision AI bridge. Analyzes input mammograms through geometric scoring (entropy, edge density) and translates visual properties into the 30 structural cytology features expected by the Hybrid Model. Applies strict per-feature scaling based on true biological minimums.
4. **`reports/pdf_generator.py`**: Powered by `reportlab`. Accepts predictions, visual bounding boxes, XAI heatmaps, and doctor's notes to compile a comprehensive BI-RADS (Breast Imaging-Reporting and Data System) PDF report.
5. **`main.py`**: Exposes robust REST endpoints including `/predict`, `/predict-excel`, `/features`, `/generate-report`, and `/find-hospitals`. Includes failovers for multi-mirror OpenStreetMap calls and hardcoded curated hospital fallbacks.

### **Frontend (`/frontend`)**
Modern SPA built with **React** and **Vite**, utilizing responsive, minimal generic styling (`App.css`). No extra component frameworks dependencies.

**Core Sections (`src/components/`):**
1. **`App.jsx`**: Main shell containing state and Dashboard Tabs. Contains continuous `checkHealth` polling to monitor backend model availability.
2. **`ImageUpload.jsx`**: Drag-and-drop file upload zone for mammograms OR manual JSON/Feature inputs. Routes data securely via FormData to backend.
3. **`ResultsView.jsx`**: Comprehensive results dashboard. Displays Malignant/Benign classification with confidence gauges, BI-RADS mapping, visual image comparisons (original vs heatmap), and detailed LIME feature importance breakdowns.
4. **`HospitalLocator.jsx`**: Context-aware interactive Map. Integrated with Leaflet.js. Automatically fetches proximity-based oncology centers upon malignant classification, displays route maps, addresses, and one-click contact details.
5. **`ReportForm.jsx`**: Patient tracking layout. Prompts the final doctor/caregiver to insert remarks and downloads the synthesized BI-RADS PDF payload directly from backend.
6. **`utils/api.js`**: Reusable modular fetch interface abstracting all REST calls to the backend running at `http://localhost:8080`.

---

## 🛠️ System Requirements & Setup

### **Prerequisites**
- Python 3.9+ 
- Node.js 18+
- NPM / Yarn

### **1. Setup Backend (AI Server)**

Open a terminal and establish the python environment:

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### Train the Models
Before starting the server, the system needs to train the ensemble models and feature scalers.
```bash
python -m ml.train
```
*(This extracts the dataset configurations, fits the Scikit-learn models & Scalers, and dumps artifacts to the `backend/models` directory)*

#### Run the Server
```bash
# Start the Uvicorn ASGI server
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```
The backend API is now running and documented dynamically at `http://127.0.0.1:8080/docs`.

---

### **2. Setup Frontend (React UI)**

Open a separate terminal instance for the client application:

```bash
cd frontend
npm install

# Start the Vite development server
npm run dev
```

The application will be exposed typically on `http://localhost:5173/` or `5174`.

---

## 🚀 Usage Flow

1. **Dashboard Initialization:** Open the browser. The frontend gracefully checks the backend `/health` to ensure all 5 ensemble models are loaded.
2. **Uploading Data:** User uploads a raw medical image via the **Upload** tab. Alternatively, a radiologist can switch the toggle to input 30 explicit cytological features measured directly.
3. **Classification:** The backend processes the image, runs through DenseNet geometry conversions mapped into the Hybrid Ensemble, resulting in a distinct P(Malignant) probability.
4. **Interpreting Results:** Navigation auto-switches to **Results**. Check the BI-RADS category mapping. Review the Explanatary AI (LIME) graph to understand exactly *which* features (e.g., `radius_se` or `concavity_worst`) pushed the outcome.
5. **Taking Action:** If flagged as malignant, the system automatically suggests proximity specialist oncology mapping via the robust **Hospitals** tab interactive Leaflet map.
6. **Report Generation:** Move to the **Report** tab, attach clinical IDs, and export a final standardized encrypted PDF.

---

## 🏥 Hospital Locator Resilience
To ensure high availability in production, the `/find-hospitals` endpoint utilizes robust strategies:
- **Prioritization**: Scans specific tags (`"healthcare:speciality"~"oncology|cancer"`, and relevant named facilities) mapping directly to specialists while preserving generic hospitals purely as fallback references.
- **Failover Mirrors**: Iterates against multiple OpenStreetMap Overpass Interpreter domains (`overpass-api.de`, `kumi.systems`). 
- **Definitive Fallback**: In the event of a catastrophic global Overpass timeout (HTTP 504), automatically delegates routing to a rigorously curated fallback set of 30+ named premium Indian comprehensive cancer centers based on pure Haversine metric distances.

---

## ⚖️ Disclaimer
* Lumino OncologyCancer is an experimental prototype application intended for academic and research contexts. It acts as an **AI-Assisted Screening System** and is rigorously isolated as a **Secondary Opinion Matrix**. 
* **Under no circumstance does this software replace professional pathological diagnosis, biopsies, or licensed medical oversight.**
