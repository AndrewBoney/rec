from __future__ import annotations

import argparse
import importlib
from typing import Optional


def promote_best_artifact(
    *,
    project: str,
    entity: Optional[str],
    metric: str,
    artifact_name: str,
    artifact_type: str,
    alias: str,
    maximize: bool,
    run_id: Optional[str],
) -> str:
    """Promote the best run's artifact by adding an alias (e.g., production)."""
    wandb = importlib.import_module("wandb")
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    if run_id:
        best_run = api.run(f"{project_path}/{run_id}")
    else:
        runs = api.runs(project_path)
        if not runs:
            raise RuntimeError("No runs found for the specified project.")

        def _score(run) -> float:
            value = run.summary.get(metric)
            return float(value) if value is not None else float("-inf")

        best_run = max(runs, key=_score) if maximize else min(runs, key=_score)

    target_artifact = None
    for artifact in best_run.logged_artifacts():
        if artifact.type != artifact_type:
            continue
        if artifact.name.split(":")[0] == artifact_name:
            target_artifact = artifact
            break

    if target_artifact is None:
        raise RuntimeError(f"No artifact named '{artifact_name}' found on best run.")

    if alias not in target_artifact.aliases:
        target_artifact.aliases.append(alias)
        target_artifact.save()

    return target_artifact.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote the best W&B model artifact")
    parser.add_argument("--project", default="rec")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--metric", default="val/recall@10", help="Metric to rank runs by")
    parser.add_argument("--artifact-name", required=True, help="Artifact name to promote")
    parser.add_argument("--artifact-type", default="model")
    parser.add_argument("--alias", default="production")
    parser.add_argument("--run-id", default=None, help="Run id to promote (overrides best-by-metric)")
    parser.add_argument("--minimize", action="store_true", help="Minimize the metric instead of maximizing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_name = promote_best_artifact(
        project=args.project,
        entity=args.entity,
        metric=args.metric,
        artifact_name=args.artifact_name,
        artifact_type=args.artifact_type,
        alias=args.alias,
        maximize=not args.minimize,
        run_id=args.run_id,
    )
    print(f"Promoted {artifact_name} with alias '{args.alias}'.")


if __name__ == "__main__":
    main()
