# 📡 Customer Churn MLOps Platform

A production-grade machine learning operations system for predicting customer churn in telecommunications. This project demonstrates a complete MLOps pipeline with automated model training, drift monitoring, predictions serving, and interactive UI—all orchestrated with Apache Airflow and containerized with Docker.

---

## 🏗️ Tech Stack

### Core ML & Data Processing
- **Python 3.13.7** - Core language
- **scikit-learn 1.8.0** - Machine learning (RandomForestClassifier)
- **pandas 2.3.3** - Data manipulation and analysis
- **joblib** - Model serialization and persistence
- **KaggleHub 1.0.0** - Data persistence

### Web & API Services
- **FastAPI 0.129.0** - High-performance REST API for predictions
- **Uvicorn 0.41.0** - ASGI server
- **Pydantic 2.12.5** - Data validation
- **Streamlit 1.54.0** - Interactive web UI
- **Requests** - HTTP client for API communication

### Orchestration & Scheduling
- **Apache Airflow 2.7.2** - Workflow orchestration
- **PostgreSQL 13** - Database backend for Airflow

### DevOps & Containerization
- **Docker** - Container images
- **Docker Compose** - Multi-container orchestration

---

## 📁 Directory Structure

```
Customer-Churn-MLOps/
├── 📄 README.md                         # This file
├── 📄 Dockerfile                        # Container image definition
├── 📄 docker-compose.yaml               # Multi-service orchestration
├── 📄 requirements.txt                  # Python dependencies
│
├── 📂 src/                              # Core ML/Data Pipeline
│   ├── train.py                         # Model training script
│   ├── predict.py                       # Inference engine (prediction logic)
│   ├── monitor.py                       # Data drift detection (PSI scores)
│   ├── data_gen.py                      # Synthetic data generation
│   └── imp_features.py                  # Feature importance calculation
│
├── 📂 api/                              # FastAPI Prediction Server
│   └── app.py                           # REST API endpoints
│
├── 📂 ui/                               # Streamlit Frontend
│   └── app.py                           # Interactive web interface
│
├── 📂 airflow/                          # Airflow Orchestration
│   └── dags/
│       └── drift_dag.py                 # Daily monitoring & retraining pipeline
│
├── 📂 models/                           # Trained ML Artifacts
│   ├── training_model.joblib            # Trained Random Forest classifier
│   ├── encoder.joblib                   # Feature encoder
│   ├── imp_features.txt                 # List of important features
│   ├── PSI_scores.txt                   # Data drift detection scores
│   └── validation_probabilities.npy     # Cached validation predictions
│
└── 📂 data/                             # Data Storage
    ├── live_inference_logs.csv          # Prediction logs & ground truth
    └── batches/
        ├── batch_0_training.csv         # Training dataset
        ├── batch_1_production.csv       # Production batch data
        └── batch_2_production.csv       # Additional production batch
```

---

## 🏛️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CUSTOMER CHURN MLOPS PLATFORM                     │
│                        (Docker Compose Network)                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐          JSON         ┌──────────────┐
│  🎨 UI       │◄─────────────────────►│  📡 API      │◄─────────┐
│  Streamlit   │    Prediction &       │  FastAPI     │          │
│  (Port 8501) │    Customer Data      │  (Port 8000) │          │
└──────────────┘                       └──────┬───────┘          │
                                              │                  │
┌────────────────┐               Appends Data │                  │
│ 📅 ORCHESTRATOR│                            ▼                  │
│ Apache Airflow │ ┌──────────────────┐    ┌─────────────────┐   │
│ (Port 8080)    │ │ 📊 MONITORING    │◄───┤  📊 DATA LOGS   │   │ HTTP POST
│                ├►│ monitor.py       │    │ live_inference_ │   │ /reload
│ • DAG triggers │ │                  │    │  logs.csv       │   │
└───────┬────────┘ └────────┬─────────┘    └─────────────────┘   │
        │                   │                                    │
        │                   │ If drift > 0.2                     │
        │                   ▼                                    │
        │          ┌──────────────────┐    ┌───────────────────┐ │
        │          │ 🏋️ TRAINING      │    │ 📦 MODEL ARTIFACTS│ │
        └─────────►│ train.py         ├───►│ model.joblib      ├─┤ Loads
                   │                  │Save│ features.txt      │ │
                   └────────┬─────────┘    └───────────────────┘ │
                            │                                    │
                            └────────────────────────────────────┘
```

### System Components

| Component | Purpose | Technology | Port |
|-----------|---------|-----------|------|
| **Airflow Webserver** | Orchestration & monitoring UI | Apache Airflow | 8080 |
| **Airflow Scheduler** | Executes DAG tasks automatically | Apache Airflow | - |
| **FastAPI** | Prediction API server | FastAPI + Uvicorn | 8000 |
| **Streamlit** | Interactive web dashboard | Streamlit | 8501 |
| **PostgreSQL** | Stores Airflow metadata | PostgreSQL | 5432 |

### Data Flow

1. **Training:** Load Telco customer dataset → Extract important features → Train RandomForest model → Save artifacts
2. **Monitoring:** Periodically check for data drift using PSI (Population Stability Index) scores
3. **Inference:** Process customer input → Encode features → Run prediction → Log results
4. **Auto-Retraining:** If drift detected → Retrain model automatically via Airflow

---

## 🚀 How to Run Locally

### Prerequisites

Ensure you have installed:
- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **Git**

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Customer-Churn-MLOps
   ```

2. **Start all services with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - PostgreSQL database
   - Apache Airflow (webserver + scheduler)
   - FastAPI prediction server
   - Streamlit UI

3. **Access the services:**

   | Service | URL | Credentials |
   |---------|-----|-------------|
   | **Airflow UI** | http://localhost:8080 | `airflow` / `airflow` |
   | **FastAPI Docs** | http://localhost:8000/docs | N/A |
   | **Streamlit UI** | http://localhost:8501 | N/A |

### Detailed Setup Steps

#### 1️⃣ Build Docker Images
```bash
docker-compose build
```

#### 2️⃣ Initialize Airflow Database
```bash
docker-compose run airflow-init
```
Wait 30 seconds for PostgreSQL to be ready first.

#### 3️⃣ Start Services
```bash
docker-compose up -d
```

#### 4️⃣ Verify Services are Running
```bash
docker-compose ps
```

You should see all containers in "Up" state:
- `postgres` - Database
- `airflow-webserver` - Airflow UI
- `airflow-scheduler` - Task scheduler
- `fastapi` - API server
- `streamlit` - Web UI

#### 5️⃣ Check Logs (if issues)
```bash
# Airflow logs
docker-compose logs airflow-scheduler

# FastAPI logs
docker-compose logs fastapi

# Streamlit logs
docker-compose logs streamlit
```

#### 6️⃣ Train Model (One-time Setup)
First, train the model to generate artifacts:
```bash
docker-compose exec fastapi python src/train.py
```

#### 7️⃣ Trigger Airflow DAG
1. Open http://localhost:8080
2. Login with `airflow` / `airflow`
3. Find `daily_churn_retraining_pipeline` DAG
4. Click "Trigger DAG" button
5. Monitor execution in the UI

---

## 📋 Usage Examples

### Make Predictions via API

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "Contract": "One year",
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "prediction": 1,
  "status": "Prediction logged successfully"
}
```

### Use Streamlit UI
1. Navigate to http://localhost:8501
2. Adjust customer parameters with sliders and dropdowns
3. Click "Predict Churn" button
4. View probability and recommendation

### Monitor Drift & Retraining
1. Go to Airflow UI (http://localhost:8080)
2. Monitor `daily_churn_retraining_pipeline` DAG runs
3. Check PSI scores in `models/PSI_scores.txt`
4. View prediction logs in `data/live_inference_logs.csv`

---

## 🔧 Development & Customization

### Running Individual Services Locally (without Docker)

**1. Setup Python Virtual Environment:**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. Train Model:**
```bash
python src/train.py
```

**3. Start FastAPI Server:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**4. Start Streamlit App (in another terminal):**
```bash
streamlit run ui/app.py
```

### Modifying the Pipeline

- **Add features:** Edit `src/imp_features.py`
- **Change model:** Modify classifier in `src/train.py`
- **Adjust drift threshold:** Update PSI calculation in `src/monitor.py`
- **Schedule changes:** Edit `airflow/dags/drift_dag.py` schedule_interval

### Model Retraining

The model **automatically retrains daily** if drift is detected. To manually retrain:
```bash
docker-compose exec fastapi python src/train.py
```

---

## 📊 Key Metrics & Monitoring

### PSI (Population Stability Index)
Detects data drift between training and production batches. Higher PSI = more drift.

### Prediction Logs
Stored in `data/live_inference_logs.csv`:
- Customer features
- Churn probability
- Timestamp
- Ground truth (when available for model improvement)

### Airflow DAG Runs
Track in Airflow UI at http://localhost:8080

---

## 🛑 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port 8080 already in use** | Change in docker-compose.yaml: `"8081:8080"` |
| **PostgreSQL connection refused** | Wait 30s for DB to start, then run `docker-compose restart` |
| **Models not found error** | Run `docker-compose exec fastapi python src/train.py` |
| **API returns 500 error** | Check logs: `docker-compose logs fastapi` |
| **Streamlit can't connect to API** | Ensure FastAPI service is running: `docker-compose ps` |

---

## 📈 Production Deployment

For production:
1. Use environment variables for secrets (API keys, DB credentials)
2. Configure proper monitoring & logging (Prometheus, ELK stack)
3. Set up CI/CD pipeline (GitHub Actions, GitLab CI)
4. Use Kubernetes for orchestration instead of Docker Compose
5. Implement model versioning & rollback strategy
6. Add authentication/authorization to APIs
7. Monitor model performance metrics continuously

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📧 Contact & Support

For questions or issues, refer to the project documentation or create an issue in the repository.

---

**Last Updated:** February 21, 2026
