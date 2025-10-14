import os
import datetime
import docker
import uuid
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional, List

# ---> 1. IMPORTA EL MIDDLEWARE DE CORS <---
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

app = FastAPI(title="BioboxMed MLOps Orchestrator v4.0 - Dynamic Dashboard")

# ---> 2. CONFIGURA CORS <---
# Define de dónde permitimos las solicitudes. En producción, sería tu dominio real.
origins = [
    "http://localhost:3000",
]

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

# --- Endpoints ---

# ---> 3. NUEVO ENDPOINT PARA OBTENER TODOS LOS MODELOS <---
@app.get("/models", tags=["Dashboard"])
def get_all_models(db: Session = Depends(get_db)):
    """Devuelve una lista de todos los modelos registrados en la base de datos."""
    return db.query(DBModel).order_by(DBModel.created_at.desc()).all()

# ---> 4. NUEVO ENDPOINT PARA OBTENER RESUMEN DEL FEEDBACK <---
@app.get("/feedback/summary", tags=["Dashboard"])
def get_feedback_summary(db: Session = Depends(get_db)):
    """Calcula y devuelve el conteo total de aciertos y errores."""
    correct_count = db.query(DBFeedback).filter(DBFeedback.is_correct == True).count()
    incorrect_count = db.query(DBFeedback).filter(DBFeedback.is_correct == False).count()
    return {"correct": correct_count, "incorrect": incorrect_count}

# ... (El resto de tus endpoints: /training-jobs, /predict, etc., permanecen igual) ...
@app.post("/training-jobs", status_code=202, tags=["MLOps Lifecycle"])
def create_training_job(job: TrainingJob, db: Session = Depends(get_db)):
    model_id_str = f"{job.model_name}_v{datetime.date.today().strftime('%Y-%m-%d')}"
    
    if db.query(DBModel).filter(DBModel.model_id == model_id_str).first():
        raise HTTPException(status_code=409, detail=f"Conflict: El modelo '{model_id_str}' ya fue procesado hoy.")

    db_model = DBModel(model_id=model_id_str, dataset_id=job.dataset_id, status="PENDING")
    db.add(db_model)
    db.commit()
    
    print(f"Lanzando worker para entrenar el modelo: {model_id_str}")
    try:
        worker_image_name = "bioboxmed-mlops-prototype-ml_worker"
        
        environment_vars = {
            "DATABASE_URL": DATABASE_URL
        }

        docker_client.containers.run(
            worker_image_name,
            command=[f"--model-id={model_id_str}", f"--dataset-id={job.dataset_id}"],
            detach=True, 
            network="bioboxmed-mlops-prototype_default",
            volumes_from=["biobox_backend"], 
            remove=True,
            environment=environment_vars
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
    db.commit()
    return {"message": f"Modelo {model_id} actualizado a COMPLETED."}

@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.model_id == request.model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail=f"Modelo '{request.model_id}' no encontrado.")
    if db_model.status != "COMPLETED":
        raise HTTPException(status_code=409, detail=f"El modelo '{request.model_id}' no está listo. Estado actual: {db_model.status}")
    
    try:
        model = load_model(db_model.model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {e}")
            
    try:
        image_tensor = preprocess_image(request.image_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archivo de imagen no encontrado en: {request.image_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")
            
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    class_names = ['NORMAL', 'NEUMONIA']
    prediction = class_names[predicted_idx.item()]
    
    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
    
    return {
        "prediction_id": prediction_id,
        "model_used": request.model_id,
        "prediction": prediction
    }

@app.get("/models/{model_id}", tags=["MLOps Lifecycle"])
def get_model_details(model_id: str, db: Session = Depends(get_db)):
    model = db.query(DBModel).filter(DBModel.model_id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return model

@app.post("/feedback", status_code=201, tags=["Feedback Loop"])
def receive_feedback(feedback: Feedback, db: Session = Depends(get_db)):
    db_feedback = DBFeedback(
        prediction_id=feedback.prediction_id,
        is_correct=feedback.is_correct,
        radiologist_notes=feedback.radiologist_notes,
        model_version_id=feedback.model_version_id
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback
@app.get("/models/{model_id}/validation-report", tags=["Dashboard"])
def get_validation_report(model_id: str, db: Session = Depends(get_db)):
    """
    Genera un informe de validación para un modelo específico basado en el feedback recolectado.
    """
    # Verificar que el modelo existe
    model = db.query(DBModel).filter(DBModel.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_id}' no encontrado.")

    # Contar todo el feedback asociado a esta versión del modelo
    feedbacks = db.query(DBFeedback).filter(DBFeedback.model_version_id == model_id)
    
    total_feedback = feedbacks.count()
    if total_feedback == 0:
        return {"model_id": model_id, "message": "No hay suficiente feedback para generar un reporte."}

    correct_predictions = feedbacks.filter(DBFeedback.is_correct == True).count()
    incorrect_predictions = total_feedback - correct_predictions
    
    # Simulación de Falsos Negativos/Positivos basado en las notas
    # En un sistema real, esto podría ser más sofisticado (ej. usando etiquetas en el feedback)
    false_negatives = feedbacks.filter(
        DBFeedback.is_correct == False, 
        DBFeedback.radiologist_notes.ilike('%falso negativo%')
    ).count()

    # Calcular métricas clave de validación
    accuracy = (correct_predictions / total_feedback) * 100
    error_rate = (incorrect_predictions / total_feedback) * 100

    return {
        "model_id": model_id,
        "total_validated_predictions": total_feedback,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "real_world_accuracy_percent": round(accuracy, 2),
        "error_rate_percent": round(error_rate, 2),
        "detected_false_negatives": false_negatives,
        "generated_at": datetime.datetime.now()
    }