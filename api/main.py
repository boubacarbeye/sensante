# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvée. "
        "/explain sera désactivé."
    )



class ExplainInput(BaseModel):
    diagnostic: str = Field(...,
        description="Diagnostic prédit par le modèle")
    probabilite: float = Field(...,
        description="Probabilité du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(...,
        description="Explication en français")
    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé")

# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)


    # Autoriser les requêtes depuis le frontend
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En dev : tout accepter
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# --- Chargement du modele (une seule fois) ---
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modele charge : {list(model.classes_)}")

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", "N/A"),
        "classes": list(model.classes_),
        "n_features": len(feature_cols)
    }

SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours
une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni."""


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Clé API non configurée."
            ),
            modele_llm="aucun"
        )

    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} "
        f"(probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
        
    return ExplainOutput ( explication = explication )
    
@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encoder
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # Features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])


    # Prediction
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    confiance = (
        "haute" if proba_max >= 0.7
        else "moyenne" if proba_max >= 0.4
        else "faible"
    )

    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )

   
