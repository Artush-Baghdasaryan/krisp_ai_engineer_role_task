import asyncio
import logging
from typing import List, Tuple

from openai import AsyncOpenAI

from src.core import Config
from src.data import DataLoader
from src.models import Cluster
from src.prompts.classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
)
from src.utils import extract_json, log_usage

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(self, data_loader: DataLoader, config: Config):
        self._data_loader = data_loader
        self._question_col = config.question_column
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._model = config.llm_model
        self._concurrency = 10


    async def classify(self, clusters: List[Cluster]) -> Tuple[List[Cluster], List[str]]:
        if not clusters:
            return [], []

        cluster_dict = self.__build_cluster_dict_with_zero_counts(clusters)
        system_prompt = self.__build_system_prompt(clusters)
        chunks = self.__collect_question_chunks()

        if not chunks:
            logger.warning("No question chunks to classify")
            return sorted(cluster_dict.values(), key=lambda c: c.id), []

        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = [
            self.__classify_chunk(questions, idx + 1, system_prompt, semaphore)
            for idx, questions in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.__merge_assignments_into_counts(cluster_dict, results)
        predicted_cluster_ids = self.__collect_predicted_ids_in_order(results)

        logger.info("Classification done: %d chunks processed", len(chunks))
        return (
            sorted(cluster_dict.values(), key=lambda c: c.id),
            predicted_cluster_ids,
        )


    @staticmethod
    def __build_cluster_dict_with_zero_counts(clusters: List[Cluster]) -> dict[str, Cluster]:
        return {
            c.id: Cluster(id=c.id, name=c.name, description=c.description, count=0)
            for c in clusters
        }


    def __collect_question_chunks(self) -> List[List[str]]:
        chunks: List[List[str]] = []
        for chunk in self._data_loader.stream_dataframe():
            if chunk.empty or self._question_col not in chunk.columns:
                continue
            questions = chunk[self._question_col].astype(str).tolist()
            if questions:
                chunks.append(questions)
        return chunks


    async def __classify_chunk(
        self,
        questions: List[str],
        chunk_idx: int,
        system_prompt: str,
        semaphore: asyncio.Semaphore,
    ) -> List[str]:
        async with semaphore:
            content = await self.__fetch_classification_response(questions, system_prompt)
            if not content:
                logger.warning("Classification chunk %d: empty LLM response, skipping", chunk_idx)
                return []

            return self.__parse_cluster_ids_from_response(content, chunk_idx)


    async def __fetch_classification_response(
        self, questions: List[str], system_prompt: str
    ) -> str:
        user_content = self.__build_user_content(questions)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        log_usage(response.usage, context="classification")
        return (response.choices[0].message.content or "").strip()


    def __parse_cluster_ids_from_response(self, content: str, chunk_idx: int) -> List[str]:
        try:
            data = extract_json(content)
        except Exception as e:
            logger.warning("Classification chunk %d: invalid JSON from LLM: %s", chunk_idx, e)
            return []

        assignments = data.get("assignments", [])
        return [self.__extract_cluster_id(a) for a in assignments if isinstance(a, dict)]


    @staticmethod
    def __extract_cluster_id(assignment: dict) -> str:
        raw = assignment.get("cluster_id")
        if raw is None:
            return ""

        return str(raw).strip() or ""


    @staticmethod
    def __merge_assignments_into_counts(
        cluster_dict: dict[str, Cluster],
        results: List[List[str] | Exception],
    ) -> None:
        for cluster_ids in results:
            if isinstance(cluster_ids, Exception):
                logger.warning("Classification chunk failed: %s", cluster_ids)
                continue

            for cid in cluster_ids:
                if cid and cid in cluster_dict:
                    cluster_dict[cid].count += 1


    @staticmethod
    def __collect_predicted_ids_in_order(
        results: List[List[str] | Exception],
    ) -> List[str]:
        predicted_cluster_ids: List[str] = []
        for cluster_ids in results:
            if isinstance(cluster_ids, Exception):
                continue
            predicted_cluster_ids.extend(cluster_ids)

        return predicted_cluster_ids


    @staticmethod
    def __build_system_prompt(clusters: List[Cluster]) -> str:
        clusters_text = "\n".join(
            f"  id={c.id}: {c.name} — {c.description}" for c in clusters
        )
        return CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE.format(clusters_text=clusters_text)


    @staticmethod
    def __build_user_content(questions: List[str]) -> str:
        lines = [f"[{i}] {q}" for i, q in enumerate(questions)]
        return CLASSIFICATION_USER_PROMPT_TEMPLATE.format(questions_text="\n".join(lines))
