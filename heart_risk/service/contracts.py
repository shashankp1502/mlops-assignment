"""Pydantic request/response contracts for the heart-risk API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

VALID_THAL_CODES = {3, 6, 7}


class PatientRecord(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels")
    thal: int

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value: int) -> int:
        if value not in VALID_THAL_CODES:
            raise ValueError(f"thal must be one of {sorted(VALID_THAL_CODES)}")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 211,
                "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0,
                "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
            }
        }
    }


class RiskAssessment(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int
    risk_label: str
    confidence: float
    probability_disease: float
    model_name: str
    model_version: str


class ServiceStatus(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_ready: bool
    model_name: str | None = None
    model_version: str | None = None
