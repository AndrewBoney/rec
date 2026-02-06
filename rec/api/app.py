import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .service import RecService


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    user_id: str
    top_k: Optional[int] = None


def build_auth_dependency(tokens: List[str], require_auth: bool):
    token_set = {token.strip() for token in tokens if token.strip()}

    if require_auth and not token_set:
        raise ValueError("Authentication requested but REC_API_TOKENS/API_TOKENS is empty")
    if not token_set:
        return None

    def verify(authorization: Optional[str] = Header(default=None)) -> None:
        token = (authorization or "").replace("Bearer ", "").strip()
        if token not in token_set:
            raise HTTPException(status_code=401, detail="Unauthorized")

    return verify


def create_app(service: RecService, tokens: List[str], require_auth: bool) -> FastAPI:
    auth_dep = build_auth_dependency(tokens, require_auth)
    dependencies = [Depends(auth_dep)] if auth_dep else []
    app = FastAPI(title="rec-api", version="1.0.0", dependencies=dependencies)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            }
        )
        return response

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "healthy"}

    @app.get("/readiness")
    def readiness() -> Dict[str, Any]:
        try:
            count = service.vector_store.count()
            if count == 0:
                raise HTTPException(status_code=503, detail="Vector store is empty")
            return {
                "status": "ready",
                "num_items": service.num_items,
                "vector_store_items": count,
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

    @app.post("/predict")
    def predict(request: PredictRequest) -> Dict[str, Any]:
        return service.predict(request.user_id, request.top_k)

    return app
