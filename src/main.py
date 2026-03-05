import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core import Config, setup_logging
from src.data import DataLoader
from src.evaluation import evaluate_clustering
from src.services import (
    ClassificationService,
    ClusterService,
    DedupeService,
    EmbeddingService,
)

load_dotenv()

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    config = Config()
    config.validate()

    data_loader = DataLoader(config)
    embedding_service = EmbeddingService(config)
    dedupe_service = DedupeService(embedding_service, config)
    cluster_service = ClusterService(config)
    classification_service = ClassificationService(data_loader, config)

    async def run():
        try:
            logger.info("Loading dataset from %s", config.dataset_path)
            df = data_loader.load_dataframe()
            logger.info("Loaded %d rows", len(df))

            logger.info("Step 1: Deduplicating (exact)")
            exact_deduped = dedupe_service.exact_dedupe(df)
            logger.info("Exact deduped to %d rows", len(exact_deduped))

            logger.info("Step 1b: Deduplicating (semantic)")
            deduped = await dedupe_service.semantic_dedupe(exact_deduped)
            logger.info("Semantic deduped to %d rows", len(deduped))

            logger.info(f"{deduped}")

            logger.info("Step 2: Clustering (LLM)")
            clusters = await cluster_service.cluster(deduped)
            logger.info("Found %d clusters", len(clusters))

            logger.info(f"{clusters}")

            logger.info("Step 3: Classifying")
            clusters_with_counts, predicted_cluster_ids = await classification_service.classify(clusters)
            logger.info("Classification done")

            output_path = Path("data/output.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [c.model_dump() for c in clusters_with_counts]
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved result to %s", output_path)

            metrics = evaluate_clustering(df[config.label_column], predicted_cluster_ids)
            logger.info("Clustering evaluation: ARI=%.4f NMI=%.4f V-measure=%.4f", metrics["adjusted_rand_index"], metrics["normalized_mutual_information"], metrics["v_measure"])
            evaluation_path = Path("data/evaluation.json")
            evaluation_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            logger.info("Saved evaluation to %s", evaluation_path)

        except Exception:
            logger.exception("Pipeline failed")
            raise

    asyncio.run(run())


if __name__ == "__main__":
    main()
