from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import chromadb
import litserve as ls
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .common.data import DataPaths, FeatureStore, build_feature_store
from .common.io import load_model_from_bundle, load_model_from_wandb
from .common.utils import FeatureConfig, read_parquet_batches
from .common.utils import to_device as to_device_tensors
from .common.utils import load_config
from .ranking.model import TwoTowerRanking
from .retrieval.model import TwoTowerRetrieval


@dataclass
class LatestInteraction:
    item_id: str
    timestamp: Optional[str]


def _parse_devices(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def _build_feature_config(cfg: Dict[str, Any], stage: str) -> FeatureConfig:
    dataset = cfg.get("dataset", {})
    base_columns = dataset.get("columns", {})
    stage_columns = cfg.get(stage, {}).get("columns", {})
    columns = {**base_columns, **stage_columns}

    return FeatureConfig(
        user_id_col=columns.get("user_id", "user_id"),
        item_id_col=columns.get("item_id", "item_id"),
        user_cat_cols=columns.get("user_cat_cols", []),
        item_cat_cols=columns.get("item_cat_cols", []),
        interaction_user_col=columns.get("interaction_user_col", columns.get("user_id", "user_id")),
        interaction_item_col=columns.get("interaction_item_col", columns.get("item_id", "item_id")),
        interaction_time_col=columns.get("interaction_time_col"),
        interaction_label_col=columns.get("label_col"),
    )


def _build_data_paths(cfg: Dict[str, Any]) -> DataPaths:
    dataset = cfg.get("dataset", {})
    paths = dataset.get("paths", {})
    return DataPaths(
        users_path=paths["users"],
        items_path=paths["items"],
        interactions_train_path=paths["interactions_train"],
        interactions_val_path=paths["interactions_val"],
    )


def _load_latest_interactions(
    interactions_path: str,
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    chunksize: int,
) -> Dict[str, LatestInteraction]:
    latest: Dict[str, LatestInteraction] = {}
    latest_time: Dict[str, pd.Timestamp] = {}

    for chunk in read_parquet_batches(interactions_path, batch_size=chunksize):
        if time_col and time_col in chunk.columns:
            chunk = chunk.copy()
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
            sorted_chunk = chunk.sort_values(time_col)
        else:
            sorted_chunk = chunk

        latest_rows = sorted_chunk.groupby(user_col, as_index=False).tail(1)
        for row in latest_rows.itertuples(index=False):
            user_val = str(getattr(row, user_col))
            item_val = str(getattr(row, item_col))
            ts_val = getattr(row, time_col) if time_col and time_col in chunk.columns else None
            if time_col and ts_val is not None and not pd.isna(ts_val):
                prev = latest_time.get(user_val)
                if prev is not None and ts_val <= prev:
                    continue
                latest_time[user_val] = ts_val
                latest[user_val] = LatestInteraction(item_id=item_val, timestamp=str(ts_val.date()))
            else:
                latest[user_val] = LatestInteraction(item_id=item_val, timestamp=None)

    return latest


def _decode_user_item_list(values: Iterable[Any]) -> List[str]:
    return [str(v) for v in values]


def _build_auth_middleware(tokens: List[str]):
    token_set = {token.strip() for token in tokens if token.strip()}

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            auth_header = request.headers.get("authorization", "")
            token = auth_header.replace("Bearer ", "").strip()
            if token not in token_set:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return await call_next(request)

    return AuthMiddleware


def _build_chroma_indexes(args: argparse.Namespace, client) -> None:
    cfg = load_config(args.config)
    data_paths = _build_data_paths(cfg)

    if args.retrieval_wandb:
        retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_wandb(
            args.retrieval_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
    else:
        retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_bundle(
            args.retrieval_bundle
        )

    if args.ranking_wandb:
        ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_wandb(
            args.ranking_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
    else:
        ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_bundle(
            args.ranking_bundle
        )

    if retrieval_meta.get("feature_config"):
        feature_cfg = FeatureConfig(**retrieval_meta["feature_config"])
    else:
        feature_cfg = _build_feature_config(cfg, stage="retrieval")

    ranking_feature_cfg = ranking_meta.get("feature_config")
    if ranking_feature_cfg and ranking_feature_cfg != retrieval_meta.get("feature_config"):
        raise ValueError("Ranking and retrieval feature configs differ; ensure consistent training configs.")

    if rank_user_enc != user_encoders or rank_item_enc != item_encoders:
        raise ValueError("Ranking and retrieval encoders differ; ensure consistent artifacts.")

    feature_store = build_feature_store(data_paths, feature_cfg, user_encoders, item_encoders)

    if args.rebuild_index:
        for name in (args.chroma_collection, args.chroma_ranking_collection):
            try:
                client.delete_collection(name)
            except Exception:
                pass

    retrieval_collection = client.get_or_create_collection(
        name=args.chroma_collection,
        metadata={"hnsw:space": "ip"},
    )
    ranking_collection = client.get_or_create_collection(
        name=args.chroma_ranking_collection,
        metadata={"hnsw:space": "ip"},
    )

    device = torch.device(args.index_device)
    retrieval_model = retrieval_model.to(device)
    ranking_model = ranking_model.to(device)
    retrieval_model.eval()
    ranking_model.eval()

    item_features = feature_store.get_all_item_features()
    item_features = to_device_tensors(item_features, device)
    with torch.no_grad():
        retrieval_emb = retrieval_model.item_tower(item_features).cpu().numpy()
        ranking_emb = ranking_model.item_tower(item_features).cpu().numpy()

    encoded_item_ids = feature_store.get_all_item_ids().tolist()
    reverse_item_map = {v: k for k, v in item_encoders[feature_cfg.item_id_col].mapping.items()}
    raw_item_ids = [str(reverse_item_map.get(int(idx), idx)) for idx in encoded_item_ids]

    num_items = len(raw_item_ids)
    for start in range(0, num_items, args.chroma_batch_size):
        end = start + args.chroma_batch_size
        batch_ids = raw_item_ids[start:end]
        retrieval_collection.add(ids=batch_ids, embeddings=retrieval_emb[start:end].tolist())
        ranking_collection.add(ids=batch_ids, embeddings=ranking_emb[start:end].tolist())


def build_api(args: argparse.Namespace, collection):
    class RecAPI(ls.LitAPI):
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()
            self.args = args
            self.device = torch.device("cpu")
            self.feature_store: Optional[FeatureStore] = None
            self.latest_interactions: Dict[str, LatestInteraction] = {}
            self.collection = collection
            self.num_items = 0
            self.user_id_encoder = None
            self.item_id_encoder = None
            self.retrieval_model: Optional[TwoTowerRetrieval] = None
            self.ranking_model: Optional[TwoTowerRanking] = None

        def setup(self, device: str) -> None:
            self.device = torch.device(device)
            cfg = load_config(self.args.config)
            data_paths = _build_data_paths(cfg)

            if self.args.retrieval_wandb:
                retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_wandb(
                    self.args.retrieval_wandb,
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity,
                )
            else:
                retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_bundle(
                    self.args.retrieval_bundle
                )

            if self.args.ranking_wandb:
                ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_wandb(
                    self.args.ranking_wandb,
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity,
                )
            else:
                ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_bundle(
                    self.args.ranking_bundle
                )

            if retrieval_meta.get("feature_config"):
                feature_cfg = FeatureConfig(**retrieval_meta["feature_config"])
            else:
                feature_cfg = _build_feature_config(cfg, stage="retrieval")

            ranking_feature_cfg = ranking_meta.get("feature_config")
            if ranking_feature_cfg and ranking_feature_cfg != retrieval_meta.get("feature_config"):
                raise ValueError("Ranking and retrieval feature configs differ; ensure consistent training configs.")

            if rank_user_enc != user_encoders or rank_item_enc != item_encoders:
                raise ValueError("Ranking and retrieval encoders differ; ensure consistent artifacts.")

            self.feature_store = build_feature_store(data_paths, feature_cfg, user_encoders, item_encoders)
            self.latest_interactions = _load_latest_interactions(
                data_paths.interactions_train_path,
                feature_cfg.interaction_user_col,
                feature_cfg.interaction_item_col,
                feature_cfg.interaction_time_col,
                chunksize=self.args.interactions_chunksize,
            )

            self.user_id_encoder = user_encoders[feature_cfg.user_id_col]
            self.item_id_encoder = item_encoders[feature_cfg.item_id_col]

            retrieval_model = retrieval_model.to(self.device)
            ranking_model = ranking_model.to(self.device)
            retrieval_model.eval()
            ranking_model.eval()
            self.retrieval_model = cast(TwoTowerRetrieval, retrieval_model)
            self.ranking_model = cast(TwoTowerRanking, ranking_model)

            if self.collection is None:
                raise RuntimeError("Chroma collection not initialized")

            self.num_items = self.collection.count()
            if self.num_items == 0:
                raise RuntimeError(
                    "Chroma collection is empty. Build the index first with --build-index."
                )

        def decode_request(self, request: Dict[str, Any]) -> Tuple[str, int]:
            user_id = request.get("user_id")
            if user_id is None:
                raise ValueError("Missing required field: user_id")
            top_k = int(request.get("top_k", self.args.top_k))
            top_k = max(1, top_k)
            return str(user_id), top_k

        def predict(self, payload: Tuple[str, int]) -> Dict[str, Any]:
            user_id, top_k = payload
            if self.user_id_encoder is None or self.item_id_encoder is None:
                raise RuntimeError("Encoders not initialized")
            if self.feature_store is None:
                raise RuntimeError("Feature store not initialized")
            if self.collection is None:
                raise RuntimeError("Chroma collection not initialized")
            if self.retrieval_model is None or self.ranking_model is None:
                raise RuntimeError("Models not initialized")

            if user_id not in self.latest_interactions:
                return {"error": f"User '{user_id}' not found in interactions_train"}
            if user_id not in self.user_id_encoder.mapping:
                return {"error": f"User '{user_id}' not found in user encoders"}

            latest_interaction = self.latest_interactions[user_id]

            user_encoded = int(self.user_id_encoder.transform([user_id])[0])
            user_feats = self.feature_store.get_user_features(torch.tensor([user_encoded], dtype=torch.long))
            user_feats = to_device_tensors(user_feats, self.device)

            with torch.no_grad():
                user_emb = self.retrieval_model.user_tower(user_feats)

            top_k = min(top_k, self.num_items) if self.num_items else top_k
            query_emb = user_emb.squeeze(0).cpu().numpy().tolist()
            results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
            retrieved_ids = _decode_user_item_list(results.get("ids", [[]])[0])
            distances = results.get("distances", [[]])[0]

            if not retrieved_ids:
                return {
                    "user_id": user_id,
                    "top_k": top_k,
                    "latest_interaction": latest_interaction.__dict__,
                    "recommendations": [],
                }

            encoded_items = self.item_id_encoder.transform(retrieved_ids)
            item_ids_t = torch.tensor(encoded_items, dtype=torch.long)
            item_feats = self.feature_store.get_item_features(item_ids_t)
            item_feats = to_device_tensors(item_feats, self.device)

            batch_size = item_ids_t.size(0)
            user_feats_batch = {k: v.repeat(batch_size) for k, v in user_feats.items()}

            with torch.no_grad():
                scores = self.ranking_model(user_feats_batch, item_feats).cpu().numpy()

            order = np.argsort(scores)[::-1]
            recommendations = []
            for idx in order:
                item_id = retrieved_ids[idx]
                score = float(scores[idx])
                retrieval_distance = float(distances[idx]) if idx < len(distances) else None
                recommendations.append(
                    {
                        "item_id": item_id,
                        "score": score,
                        "retrieval_distance": retrieval_distance,
                    }
                )

            return {
                "user_id": user_id,
                "top_k": top_k,
                "latest_interaction": latest_interaction.__dict__,
                "recommendations": recommendations,
            }

        def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
            return output

    return RecAPI(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve retrieval + ranking models via LitServe")
    parser.add_argument("--config", default="config/movielens/movielens_1m_large.yaml")
    parser.add_argument("--retrieval-bundle", default="artifacts/retrieval")
    parser.add_argument("--ranking-bundle", default="artifacts/ranking")
    parser.add_argument("--retrieval-wandb", default=None)
    parser.add_argument("--ranking-wandb", default=None)
    parser.add_argument("--wandb-project", default="rec")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chroma-path", default="artifacts/chroma")
    parser.add_argument("--chroma-collection", default="items")
    parser.add_argument("--chroma-ranking-collection", default="items_ranking")
    parser.add_argument("--chroma-batch-size", type=int, default=2048)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--index-device", default="cpu")
    parser.add_argument("--interactions-chunksize", type=int, default=200_000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--num-api-servers", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--pretty-logs", action="store_true")
    parser.add_argument("--no-client-file", action="store_true")
    parser.add_argument("--require-auth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(override=False)

    token_env = os.getenv("REC_API_TOKENS") or os.getenv("API_TOKENS") or ""
    tokens = [token.strip() for token in token_env.split(",") if token.strip()]

    chroma_path = args.chroma_path
    client = chromadb.PersistentClient(path=chroma_path)

    if args.build_index:
        _build_chroma_indexes(args, client)
        return

    collection = client.get_or_create_collection(
        name=args.chroma_collection,
        metadata={"hnsw:space": "ip"},
    )

    api = build_api(args, collection)
    middlewares = []
    if tokens or args.require_auth:
        if not tokens:
            raise ValueError("Authentication requested but REC_API_TOKENS/API_TOKENS is empty")
        auth_middleware = _build_auth_middleware(tokens)
        middlewares.append((auth_middleware, {}))

    server = ls.LitServer(
        api,
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        workers_per_device=args.workers_per_device,
        middlewares=middlewares or None,
    )
    server.run(
        host=args.host,
        port=args.port,
        num_api_servers=args.num_api_servers,
        log_level=args.log_level,
        pretty_logs=args.pretty_logs,
        generate_client_file=not args.no_client_file,
    )


if __name__ == "__main__":
    main()
