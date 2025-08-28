from fastapi import APIRouter

router = APIRouter(
    prefix='/health',
    tags=['health']
)

@router.get("/live")
async def live():
    return {"status": "ok"}




@router.get("/ready")
async def ready():
# If you want deeper health checks (e.g., ping Milvus/ES/Mongo), do it here
    return {"status": "ready"}


