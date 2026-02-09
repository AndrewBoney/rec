# rec

Recommendation systems are hard. They require complex dataset construction from diverse sources, multiple training schedules, frequent mismatches between training objectives and deployment KPIs, custom model architectures, and versioning across multiple dimensions (data, code, and model weights), among other challenges.

Because of this complexity, it’s surprisingly difficult to find simple, end-to-end implementations that researchers can follow. Instead, many practitioners end up stitching together systems from tutorials that only cover isolated parts of the recommendation workflow. This often leads to fragmented understanding and brittle implementations.

This repository is designed to help address that gap by providing a **complete, end-to-end recommendation system**, covering data preparation, training, and deployment in a single, coherent example.

The project is inspired by Andrej Karpathy’s [nanochat](https://github.com/karpathy/nanochat), which aims to be the minimal viable implementation of a modern LLM. In the same spirit, this repo provides a **lightweight recommendation system** that does only what is necessary—while remaining simple enough for researchers to understand, modify, and extend.

__What this *is*__

- A learning resource for researchers who want to understand the full recommendation system workflow, end to end.
- A collection of clean, reusable implementations of the trickier parts of building recommendation systems.
- A foundation for experimenting with, and contributing, new or interesting ideas in the recommender systems space.

__What this *isn’t*__

- An exhaustive survey or implementation of all recommendation approaches. The focus here is deliberately narrow, primarily on **dual-encoder ('two-tower') architectures**.
- A fully production-ready system. While the code aims to be well-written, tested, and reusable, the primary goal is **clarity and minimalism**. The target is to keep the project under ~5,000 lines of readable Python. A real-world production system will be orders of magnitude larger and span multiple languages and services.

## Quick start (end-to-end)

1) Install dependencies: pip install -r requirements.txt
2) Generate a dummy dataset:
	python rec/data_prep/dummy.py --output-dir data/dummy
3) Train retrieval + ranking (uses the dummy config):
	python rec/train_all.py --config config/dummy/dummy_small.yaml
4) Build a local Chroma index (one-time):
	python rec/deploy_api.py --config config/dummy/dummy_small.yaml --retrieval-bundle artifacts/retrieval --ranking-bundle artifacts/ranking --build-index
5) Serve the API:
	python rec/deploy_api.py --config config/dummy/dummy_small.yaml --retrieval-bundle artifacts/retrieval --ranking-bundle artifacts/ranking

If you want Weights & Biases logging, set $WANDB_API_KEY and keep use_wandb enabled in the config. To disable it, set use_wandb: false in the config.

## Stage 1: Training (retrieval)

The retrieval model learns user and item embeddings for fast candidate generation.

- Entry point: rec/retrieval/train.py
- Orchestrator (retrieval + ranking): rec/train_all.py
- Configs: config/dummy/dummy_small.yaml, config/movielens/*

Outputs:
- Checkpoint: retrieval.ckpt (default)
- Inference bundle: artifacts/retrieval (default) with encoders + metadata
- W&B model artifact (optional): retrieval_bundle (default artifact name)

Run retrieval-only:
python rec/retrieval/train.py --config config/dummy/dummy_small.yaml

Override the retrieval architecture (default: two_tower):
python rec/retrieval/train.py --config config/dummy/dummy_small.yaml --model-arch two_tower

## Stage 2: Training (ranking)

The ranking model re-scores candidates from retrieval with a pointwise objective.

- Entry point: rec/ranking/train.py
- Orchestrator (retrieval + ranking): rec/train_all.py

Outputs:
- Checkpoint: ranking.ckpt (default)
- Inference bundle: artifacts/ranking (default)
- W&B model artifact (optional): ranking_bundle (default artifact name)

Run ranking-only (assumes a retrieval checkpoint):
python rec/ranking/train.py --config config/dummy/dummy_small.yaml --init-from-retrieval retrieval.ckpt

Try an alternate ranking architecture:
python rec/ranking/train.py --config config/dummy/dummy_small.yaml --model-arch dlrm

## Stage 3: Promote a model (W&B)

Use rec/wandb_promote.py to add a stable alias (for example, production) to the best model artifact based on a metric.

Example (promote retrieval):
python rec/wandb_promote.py --artifact-name retrieval_bundle --metric val/recall@10 --alias production

Example (promote ranking):
python rec/wandb_promote.py --artifact-name ranking_bundle --metric val/ndcg@10 --alias production

If you already know a run id, pass --run-id to promote that run’s artifact directly.

## Stage 4: Deploy an API (FastAPI)

Use [rec/deploy_api.py](rec/deploy_api.py) to serve a retrieval-then-ranking API with FastAPI. It loads retrieval + ranking bundles (local or W&B artifacts), builds a local ChromaDB index on startup (or uses an existing one), and serves recommendations for a user_id.

Local bundles:
python rec/deploy_api.py --config config/dummy/dummy_small.yaml --retrieval-bundle artifacts/retrieval --ranking-bundle artifacts/ranking

W&B bundles (artifact references like entity/project/retrieval_bundle:production):
python rec/deploy_api.py --config config/dummy/dummy_small.yaml --retrieval-wandb <artifact-ref> --ranking-wandb <artifact-ref>

Notes:
- Requires the fastapi, uvicorn, and chromadb Python packages.
- Authentication tokens are read from .env via REC_API_TOKENS (comma-separated) or API_TOKENS. Use --require-auth to enforce auth even if no tokens are present.
- Configure local vs remote hosting by setting --host and --port.