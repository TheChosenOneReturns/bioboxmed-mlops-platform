import os
import datetime
import docker
import uuid
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, func, JSON
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

# --- Lógica de Modelo de IA ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model(model_path: str):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# --- Configuración de la Base de Datos ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bioboxmed:supersecret@db:5432/bioboxmed_mlops")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Modelos de la Base de Datos ---
class DBModel(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="PENDING")
    dataset_id = Column(String)
    model_path = Column(String, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)

class DBFeedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, index=True)
    is_correct = Column(Boolean)
    radiologist_notes = Column(String, nullable=True)
    received_at = Column(DateTime, default=datetime.datetime.utcnow)
    model_version_id = Column(String, nullable=True)

## ---> MEJORA 1: TABLA DE AUDITORÍA <--- ##
# Basado en la Fase 1.1 del plan de mejoras [cite: 89, 90]
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, nullable=False)
    action_type = Column(String)
    entity_type = Column(String)
    entity_id = Column(String)
    details = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# --- Modelos Pydantic ---
class Feedback(BaseModel):
    prediction_id: str
    is_correct: bool
    radiologist_notes: Optional[str] = None
    model_version_id: Optional[str] = None

class TrainingJob(BaseModel):
    dataset_id: str
    model_name: str
    
class PredictionRequest(BaseModel):
    model_id: str
    image_path: str

class ModelMetrics(BaseModel):
    accuracy: float
    sensitivity: float
    specificity: float

class TrainingCompletionPayload(BaseModel):
    metrics: ModelMetrics
    model_path: str

app = FastAPI(title="BioboxMed MLOps Orchestrator v5.0 - Secure & Auditable")

# --- Configuración de CORS ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

docker_client = docker.from_env()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

## ---> MEJORA 2: CONTROL DE ACCESO BASADO EN ROLES (RBAC) <--- ##
# Basado en la Fase 1.3 del plan de mejoras [cite: 89, 93]
ROLES = {
    "admin": ["train_model", "view_report", "submit_feedback", "view_prediction", "view_audit"],
    "data_scientist": ["train_model", "view_report", "view_prediction"],
    "radiologist": ["submit_feedback", "view_prediction"],
    "auditor": ["view_audit", "view_report"]
}

def check_permission(required_permission: str):
    def dependency(user_role: str = Header("data_scientist", alias="X-User-Role")):
        # En un sistema real, el 'user_role' vendría de un token de autenticación (JWT).
        if user_role not in ROLES or required_permission not in ROLES[user_role]:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: Role '{user_role}' does not have the '{required_permission}' permission."
            )
        return user_role
    return dependency

# --- Endpoints ---

@app.get("/models", tags=["Dashboard"])
def get_all_models(db: Session = Depends(get_db)):
    return db.query(DBModel).order_by(DBModel.created_at.desc()).all()

@app.get("/feedback/summary", tags=["Dashboard"])
def get_feedback_summary(db: Session = Depends(get_db)):
    correct_count = db.query(DBFeedback).filter(DBFeedback.is_correct == True).count()
    incorrect_count = db.query(DBFeedback).filter(DBFeedback.is_correct == False).count()
    return {"correct": correct_count, "incorrect": incorrect_count}

@app.post("/training-jobs", status_code=202, tags=["MLOps Lifecycle"])
def create_training_job(job: TrainingJob, db: Session = Depends(get_db), user_role: str = Depends(check_permission("train_model"))):
    model_id_str = f"{job.model_name}_v{datetime.date.today().strftime('%Y-%m-%d')}"
    
    if db.query(DBModel).filter(DBModel.model_id == model_id_str).first():
        raise HTTPException(status_code=409, detail=f"Conflict: El modelo '{model_id_str}' ya fue procesado hoy.")

    db_model = DBModel(model_id=model_id_str, dataset_id=job.dataset_id, status="PENDING")
    db.add(db_model)
    
    ## ---> LOG DE AUDITORÍA <--- ##
    audit_log = AuditLog(
        user_id=user_role,
        action_type="MODEL_TRAINING_STARTED",
        entity_type="MODEL",
        entity_id=model_id_str,
        details=f"Training initiated for dataset '{job.dataset_id}'."
    )
    db.add(audit_log)

    db.commit()
    
    print(f"Lanzando worker para entrenar el modelo: {model_id_str}")
    try:
        worker_image_name = "bioboxmed-mlops-prototype-ml_worker"
        environment_vars = {"DATABASE_URL": DATABASE_URL}
        docker_client.containers.run(
            worker_image_name,
            command=[f"--model-id={model_id_str}", f"--dataset-id={job.dataset_id}"],
            detach=True, network="bioboxmed-mlops-prototype_default",
            volumes_from=["biobox_backend"], remove=True, environment=environment_vars
        )
    except Exception as e:
        db_model.status = "FAILED"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Error al lanzar el worker: {e}")

    return {"message": "Trabajo de entrenamiento aceptado.", "model_id": model_id_str, "status": "PENDING"}

@app.patch("/training-jobs/{model_id}/complete", tags=["MLOps Lifecycle"])
def mark_training_as_complete(model_id: str, payload: TrainingCompletionPayload, db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.model_id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado para actualizar.")
    
    db_model.status = "COMPLETED"
    db_model.model_path = payload.model_path
    db_model.accuracy = payload.metrics.accuracy
    db_model.sensitivity = payload.metrics.sensitivity
    db_model.specificity = payload.metrics.specificity

    ## ---> LOG DE AUDITORÍA <--- ##
    audit_log = AuditLog(
        user_id="ml_worker",
        action_type="MODEL_TRAINING_COMPLETED",
        entity_type="MODEL",
        entity_id=model_id,
        details=f"Training completed. Metrics: {payload.metrics.dict()}"
    )
    db.add(audit_log)
    db.commit()
    return {"message": f"Modelo {model_id} actualizado a COMPLETED."}

@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest, db: Session = Depends(get_db), user_role: str = Depends(check_permission("view_prediction"))):
    db_model = db.query(DBModel).filter(DBModel.model_id == request.model_id).first()
    if not db_model or db_model.status != "COMPLETED":
        raise HTTPException(status_code=404, detail=f"Modelo '{request.model_id}' no está disponible para predicción.")
    
    # ... (lógica de predicción sin cambios)
    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
    
    ## ---> LOG DE AUDITORÍA <--- ##
    audit_log = AuditLog(
        user_id=user_role,
        action_type="PREDICTION_GENERATED",
        entity_type="PREDICTION",
        entity_id=prediction_id,
        details=f"Prediction made using model '{request.model_id}' on image '{request.image_path}'."
    )
    db.add(audit_log)
    db.commit()
    
    return { "prediction_id": prediction_id, "model_used": request.model_id, "prediction": "NORMAL" } # Simplificado para el ejemplo

@app.post("/feedback", status_code=201, tags=["Feedback Loop"])
def receive_feedback(feedback: Feedback, db: Session = Depends(get_db), user_role: str = Depends(check_permission("submit_feedback"))):
    db_feedback = DBFeedback(
        prediction_id=feedback.prediction_id,
        is_correct=feedback.is_correct,
        radiologist_notes=feedback.radiologist_notes,
        model_version_id=feedback.model_version_id
    )
    db.add(db_feedback)
    
    ## ---> LOG DE AUDITORÍA <--- ##
    audit_log = AuditLog(
        user_id=user_role,
        action_type="FEEDBACK_SUBMITTED",
        entity_type="PREDICTION",
        entity_id=feedback.prediction_id,
        details=f"Feedback submitted as correct: {feedback.is_correct}. Notes: '{feedback.radiologist_notes}'"
    )
    db.add(audit_log)

    db.commit()
    db.refresh(db_feedback)
    return db_feedback

@app.get("/audit-logs", tags=["Auditoría"], dependencies=[Depends(check_permission("view_audit"))])
def get_audit_logs(db: Session = Depends(get_db)):
    """Endpoint protegido para que solo auditores y admins vean los logs."""
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()

@app.get("/models/{model_id}/check-performance", tags=["Monitoring"])
def check_performance_degradation(model_id: str, db: Session = Depends(get_db)):
    """
    Simula una tarea periódica que verifica la degradación del rendimiento de un modelo.
    """
    # 1. Obtener el rendimiento base del modelo cuando fue entrenado
    model = db.query(DBModel).filter(DBModel.model_id == model_id).first()
    if not model or not model.accuracy:
        raise HTTPException(status_code=404, detail=f"No se encontró un baseline de performance para el modelo '{model_id}'.")
    
    baseline_accuracy = model.accuracy

    # 2. Calcular el rendimiento reciente basado en el último feedback
    recent_feedback = db.query(DBFeedback).filter(DBFeedback.model_version_id == model_id).order_by(DBFeedback.received_at.desc()).limit(100).all()
    
    if len(recent_feedback) < 20: # Un mínimo de 20 feedbacks para ser estadísticamente relevante
        return {"message": "No hay suficiente feedback reciente para analizar la degradación.", "baseline_accuracy": baseline_accuracy}

    correct_count = sum(1 for f in recent_feedback if f.is_correct)
    recent_accuracy = correct_count / len(recent_feedback)
    
    # 3. Comparar y alertar si la performance ha caído
    degradation = baseline_accuracy - recent_accuracy
    
    if degradation > 0.05: # Umbral de degradación del 5%
        print(f"--- [ALERTA 🟠] DEGRADACIÓN DE PERFORMANCE DETECTADA en modelo '{model_id}' ---")
        print(f"    Precisión de baseline: {baseline_accuracy:.2%}")
        print(f"    Precisión reciente (últimos {len(recent_feedback)} feedbacks): {recent_accuracy:.2%}")
        print(f"    Caída de {degradation:.2%}")
        return {
            "alert": "PERFORMANCE_DEGRADATION_DETECTED",
            "model_id": model_id,
            "baseline_accuracy": baseline_accuracy,
            "recent_accuracy": recent_accuracy,
            "degradation": degradation
        }
    
    print(f"--- [OK ✅] Performance del modelo '{model_id}' estable. ---")
    return {
        "message": "Performance estable.",
        "model_id": model_id,
        "baseline_accuracy": baseline_accuracy,
        "recent_accuracy": recent_accuracy
    }