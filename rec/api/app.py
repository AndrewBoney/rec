from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .service import RecService


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

    @app.post("/predict")
    def predict(request: PredictRequest) -> Dict[str, Any]:
        return service.predict(request.user_id, request.top_k)

    return app
