from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..common.utils import to_device as to_device_tensors
from ..ranking.model import TwoTowerRanking
from ..retrieval.model import TwoTowerRetrieval
from .assets import LatestInteraction, load_model_assets
from .vector_store import ChromaVectorStore


@dataclass
class RecService:
    feature_store: Any
    latest_interactions: Dict[str, LatestInteraction]
    user_id_encoder: Any
    item_id_encoder: Any
    retrieval_model: TwoTowerRetrieval
    ranking_model: TwoTowerRanking
    vector_store: ChromaVectorStore
    device: torch.device
    top_k_default: int
    num_items: int

    @classmethod
    def from_args(cls, args) -> "RecService":
        assets = load_model_assets(args, load_interactions=True)

        device = torch.device(args.device)
        retrieval_model = assets.retrieval_model.to(device)
        ranking_model = assets.ranking_model.to(device)
        retrieval_model.eval()
        ranking_model.eval()

        vector_store = ChromaVectorStore.open(args.chroma_path, args.chroma_collection)
        num_items = vector_store.count()
        if num_items == 0:
            raise RuntimeError("Chroma collection is empty. Build the index first with --build-index.")

        user_id_encoder = assets.user_encoders[assets.feature_cfg.user_id_col]
        item_id_encoder = assets.item_encoders[assets.feature_cfg.item_id_col]

        return cls(
            feature_store=assets.feature_store,
            latest_interactions=assets.latest_interactions,
            user_id_encoder=user_id_encoder,
            item_id_encoder=item_id_encoder,
            retrieval_model=retrieval_model,
            ranking_model=ranking_model,
            vector_store=vector_store,
            device=device,
            top_k_default=args.top_k,
            num_items=num_items,
        )

    def predict(self, user_id: str, top_k: Optional[int]) -> Dict[str, Any]:
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

        resolved_top_k = int(top_k) if top_k is not None else self.top_k_default
        resolved_top_k = max(1, resolved_top_k)
        resolved_top_k = min(resolved_top_k, self.num_items) if self.num_items else resolved_top_k

        query_emb = user_emb.squeeze(0).cpu().numpy().tolist()
        results = self.vector_store.query(query_emb, resolved_top_k)

        if not results.ids:
            return {
                "user_id": user_id,
                "top_k": resolved_top_k,
                "latest_interaction": latest_interaction.__dict__,
                "recommendations": [],
            }

        encoded_items = self.item_id_encoder.transform(results.ids)
        item_ids_t = torch.tensor(encoded_items, dtype=torch.long)
        item_feats = self.feature_store.get_item_features(item_ids_t)
        item_feats = to_device_tensors(item_feats, self.device)

        batch_size = item_ids_t.size(0)
        user_feats_batch = {
            k: v.repeat((batch_size,) + (1,) * (v.dim() - 1)) for k, v in user_feats.items()
        }

        with torch.no_grad():
            scores = self.ranking_model(user_feats_batch, item_feats).cpu().numpy().reshape(-1)

        order = np.argsort(scores)[::-1]
        recommendations = []
        for idx in order:
            item_id = results.ids[idx]
            score = float(scores[idx])
            retrieval_distance = (
                float(results.distances[idx]) if idx < len(results.distances) else None
            )
            recommendations.append(
                {
                    "item_id": item_id,
                    "score": score,
                    "retrieval_distance": retrieval_distance,
                }
            )

        return {
            "user_id": user_id,
            "top_k": resolved_top_k,
            "latest_interaction": latest_interaction.__dict__,
            "recommendations": recommendations,
        }
