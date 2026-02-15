from dataclasses import dataclass
from typing import Any, Dict, List

import chromadb
import torch

from ..common.data import FeatureConfig, to_device as to_device_tensors


@dataclass(frozen=True)
class ChromaIndexConfig:
    path: str
    retrieval_collection: str
    batch_size: int
    rebuild: bool
    device: str


@dataclass
class ChromaQueryResult:
    ids: List[str]
    distances: List[float]


class ChromaVectorStore:
    def __init__(self, collection) -> None:
        self.collection = collection

    @classmethod
    def open(cls, path: str, collection_name: str) -> "ChromaVectorStore":
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "ip"},
        )
        return cls(collection)

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding: List[float], top_k: int) -> ChromaQueryResult:
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = [str(v) for v in results.get("ids", [[]])[0]]
        distances = results.get("distances", [[]])[0]
        return ChromaQueryResult(ids=ids, distances=distances)


def build_chroma_indexes(
    config: ChromaIndexConfig,
    feature_store,
    feature_cfg: FeatureConfig,
    item_encoders: Dict[str, Any],
    retrieval_model,
) -> None:
    client = chromadb.PersistentClient(path=config.path)

    if config.rebuild:
        try:
            client.delete_collection(config.retrieval_collection)
        except Exception:
            pass

    retrieval_collection = client.get_or_create_collection(
        name=config.retrieval_collection,
        metadata={"hnsw:space": "ip"},
    )
    device = torch.device(config.device)
    retrieval_model = retrieval_model.to(device)
    retrieval_model.eval()

    item_features = feature_store.get_all_item_features()
    item_features = to_device_tensors(item_features, device)
    with torch.no_grad():
        retrieval_emb = retrieval_model.item_tower(item_features).cpu().numpy()

    encoded_item_ids = feature_store.get_all_item_ids().tolist()
    reverse_item_map = {v: k for k, v in item_encoders[feature_cfg.item_id_col].mapping.items()}
    raw_item_ids = [str(reverse_item_map.get(int(idx), idx)) for idx in encoded_item_ids]

    num_items = len(raw_item_ids)
    for start in range(0, num_items, config.batch_size):
        end = start + config.batch_size
        batch_ids = raw_item_ids[start:end]
        retrieval_collection.add(ids=batch_ids, embeddings=retrieval_emb[start:end].tolist())
