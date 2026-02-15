import os

from dotenv import load_dotenv
import uvicorn

from .api.app import create_app
from .api.assets import load_model_assets
from .api.config import parse_args
from .api.service import RecService
from .api.vector_store import ChromaIndexConfig, build_chroma_indexes


def main() -> None:
    args = parse_args()
    load_dotenv(override=False)

    token_env = os.getenv("REC_API_TOKENS") or os.getenv("API_TOKENS") or ""
    tokens = [token.strip() for token in token_env.split(",") if token.strip()]

    if args.build_index:
        assets = load_model_assets(args, load_interactions=False)
        index_config = ChromaIndexConfig(
            path=args.chroma_path,
            retrieval_collection=args.chroma_collection,
            batch_size=args.chroma_batch_size,
            rebuild=args.rebuild_index,
            device=args.index_device,
        )
        build_chroma_indexes(
            index_config,
            assets.feature_store,
            assets.feature_cfg,
            assets.item_encoders,
            assets.retrieval_model,
        )
        return

    service = RecService.from_args(args)
    app = create_app(service, tokens, args.require_auth)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
