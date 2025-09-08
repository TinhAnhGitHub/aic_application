from fastapi import APIRouter

router = APIRouter(
    prefix='/health',
    tags=['health']
)

@router.get(
    "/live",
    summary="Liveness probe",
    description="Returns ok if the API process is running.",
)
async def live():
    return {"status": "ok"}




@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Reports when the service is ready to accept traffic. "
        "If you need deeper checks (Milvus/Elasticsearch/Mongo), implement them here."
    ),
)
async def ready():
# If you want deeper health checks (e.g., ping Milvus/ES/Mongo), do it here
    return {"status": "ready"}

