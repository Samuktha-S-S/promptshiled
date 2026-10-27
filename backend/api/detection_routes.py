"""
PromptShield - Detection API Endpoint

POST /api/detect   run the classifier and decision policy only (never calls the LLM)
"""

from fastapi import APIRouter, Request

from backend.schemas.detection_schema import DetectRequest, DetectResponse

router = APIRouter(prefix="/api")


@router.post("/detect", response_model=DetectResponse)
def detect(body: DetectRequest, request: Request):
    return request.app.state.gate.process_detect(body.message)
