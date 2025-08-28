from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.core.dependencies import lifespan
from app.api.health import router as health_router
from app.api.search import router as search_router

app = FastAPI(title="Hotspot Search API", version="0.1.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(search_router)


@app.get("/")
async def root():
    return {"name": app.title, "version": app.version}



if __name__ == "__main__":
    import os
    import uvicorn
    reload = os.getenv("RELOAD", "true").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
    )
