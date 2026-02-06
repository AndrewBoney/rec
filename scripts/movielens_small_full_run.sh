#!/usr/bin/env bash
set -euo pipefail

# Train retrieval + ranking on MovieLens, optionally promote W&B artifacts,
# then build the Chroma index and serve the API.
#
# Usage:
#   ./scripts/train_promote_deploy.sh [config_path]
#
# Env overrides:
#   USE_WANDB=true|false (default: true)
#   WANDB_PROJECT=rec
#   WANDB_ENTITY=
#   WANDB_ALIAS=production
#   RETRIEVAL_ARTIFACT_NAME=retrieval_bundle
#   RANKING_ARTIFACT_NAME=ranking_bundle
#   HOST=0.0.0.0
#   PORT=8000
#   BUILD_INDEX=true|false (default: true)
#   REBUILD_INDEX=true|false (default: false)
#   RETRIEVAL_BUNDLE=artifacts/retrieval
#   RANKING_BUNDLE=artifacts/ranking
#

CONFIG_PATH="${1:-config/movielens/movielens_1m_small.yaml}"

USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-rec}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ALIAS="${WANDB_ALIAS:-production}"
RETRIEVAL_ARTIFACT_NAME="${RETRIEVAL_ARTIFACT_NAME:-retrieval_bundle}"
RANKING_ARTIFACT_NAME="${RANKING_ARTIFACT_NAME:-ranking_bundle}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BUILD_INDEX="${BUILD_INDEX:-true}"
REBUILD_INDEX="${REBUILD_INDEX:-false}"

RETRIEVAL_BUNDLE="${RETRIEVAL_BUNDLE:-artifacts/retrieval}"
RANKING_BUNDLE="${RANKING_BUNDLE:-artifacts/ranking}"

if [[ -f ".env" ]]; then
  # Load environment variables from .env (including WANDB_API_KEY)
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${USE_WANDB}" == "true" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not set; skipping W&B promotion and using local bundles."
  USE_WANDB="false"
fi

echo "==> Training retrieval + ranking (config: ${CONFIG_PATH})"
python -m rec.train_all --config "${CONFIG_PATH}"

if [[ "${USE_WANDB}" == "true" ]]; then
  echo "==> Promoting retrieval artifact (${RETRIEVAL_ARTIFACT_NAME})"
  python -m rec.wandb_promote \
    --project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--entity "${WANDB_ENTITY}"} \
    --artifact-name "${RETRIEVAL_ARTIFACT_NAME}" \
    --metric "val/recall@10" \
    --alias "${WANDB_ALIAS}"

  echo "==> Promoting ranking artifact (${RANKING_ARTIFACT_NAME})"
  python -m rec.wandb_promote \
    --project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--entity "${WANDB_ENTITY}"} \
    --artifact-name "${RANKING_ARTIFACT_NAME}" \
    --metric "val/ndcg@10" \
    --alias "${WANDB_ALIAS}"
fi

if [[ "${BUILD_INDEX}" == "true" ]]; then
  echo "==> Building Chroma index"
  BUILD_ARGS=(--build-index)
  if [[ "${REBUILD_INDEX}" == "true" ]]; then
    BUILD_ARGS+=(--rebuild-index)
  fi

  if [[ "${USE_WANDB}" == "true" ]]; then
    if [[ -n "${WANDB_ENTITY}" ]]; then
      RETRIEVAL_REF="${WANDB_ENTITY}/${WANDB_PROJECT}/${RETRIEVAL_ARTIFACT_NAME}:${WANDB_ALIAS}"
      RANKING_REF="${WANDB_ENTITY}/${WANDB_PROJECT}/${RANKING_ARTIFACT_NAME}:${WANDB_ALIAS}"
    else
      RETRIEVAL_REF="${WANDB_PROJECT}/${RETRIEVAL_ARTIFACT_NAME}:${WANDB_ALIAS}"
      RANKING_REF="${WANDB_PROJECT}/${RANKING_ARTIFACT_NAME}:${WANDB_ALIAS}"
    fi

    python -m rec.deploy_api \
      --config "${CONFIG_PATH}" \
      --retrieval-wandb "${RETRIEVAL_REF}" \
      --ranking-wandb "${RANKING_REF}" \
      "${BUILD_ARGS[@]}"
  else
    python -m rec.deploy_api \
      --config "${CONFIG_PATH}" \
      --retrieval-bundle "${RETRIEVAL_BUNDLE}" \
      --ranking-bundle "${RANKING_BUNDLE}" \
      "${BUILD_ARGS[@]}"
  fi
fi

echo "==> Starting API server"
if [[ "${USE_WANDB}" == "true" ]]; then
  if [[ -n "${WANDB_ENTITY}" ]]; then
    RETRIEVAL_REF="${WANDB_ENTITY}/${WANDB_PROJECT}/${RETRIEVAL_ARTIFACT_NAME}:${WANDB_ALIAS}"
    RANKING_REF="${WANDB_ENTITY}/${WANDB_PROJECT}/${RANKING_ARTIFACT_NAME}:${WANDB_ALIAS}"
  else
    RETRIEVAL_REF="${WANDB_PROJECT}/${RETRIEVAL_ARTIFACT_NAME}:${WANDB_ALIAS}"
    RANKING_REF="${WANDB_PROJECT}/${RANKING_ARTIFACT_NAME}:${WANDB_ALIAS}"
  fi

  python -m rec.deploy_api \
    --config "${CONFIG_PATH}" \
    --retrieval-wandb "${RETRIEVAL_REF}" \
    --ranking-wandb "${RANKING_REF}" \
    --host "${HOST}" \
    --port "${PORT}"
else
  python -m rec.deploy_api \
    --config "${CONFIG_PATH}" \
    --retrieval-bundle "${RETRIEVAL_BUNDLE}" \
    --ranking-bundle "${RANKING_BUNDLE}" \
    --host "${HOST}" \
    --port "${PORT}"
fi
