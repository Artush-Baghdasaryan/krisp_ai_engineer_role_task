import logging
import random
from typing import List

import pandas as pd
from openai import AsyncOpenAI

from src.core import Config
from src.models import Cluster
from src.prompts import CLUSTER_SYSTEM_PROMPT_TEMPLATE, CLUSTER_USER_PROMPT_TEMPLATE
from src.utils import extract_json, log_usage

logger = logging.getLogger(__name__)


class ClusterService:
    def __init__(self, config: Config):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._question_col = config.question_column
        self._sample_size = 256


    async def cluster(self, df: pd.DataFrame) -> List[Cluster]:
        if df is None or df.empty:
            return []

        questions = self.__get_questions_for_clustering(df)
        if not questions:
            return []

        logger.info("Clustering %d questions", len(questions))
        content = await self.__call_llm(questions)
        if not content:
            logger.warning("Clustering: empty LLM response")
            return []

        try:
            clusters_data = self.__parse_clusters_from_response(content)
        except Exception as e:
            logger.error("Clustering: invalid JSON from LLM: %s", e, exc_info=True)
            raise

        result = self.__build_cluster_models(clusters_data)
        logger.info("Clustering done: %d clusters", len(result))
        return result


    def __get_questions_for_clustering(self, df: pd.DataFrame) -> List[str]:
        questions = df[self._question_col].astype(str).tolist()
        if len(questions) <= self._sample_size:
            return questions

        return random.sample(questions, self._sample_size)


    async def __call_llm(self, questions: List[str]) -> str:
        model = self._config.llm_model or "gpt-4o-mini"
        user_content = self.__build_user_content(questions)
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        log_usage(response.usage, context="clustering")
        return (response.choices[0].message.content or "").strip()


    @staticmethod
    def __parse_clusters_from_response(content: str) -> list:
        data = extract_json(content)
        raw = data.get("clusters", [])

        return raw if isinstance(raw, list) else []


    def __build_cluster_models(self, clusters_data: list) -> List[Cluster]:
        result: List[Cluster] = []
        for idx, item in enumerate(clusters_data):
            cluster = self.__item_to_cluster(idx, item)
            if cluster is not None:
                result.append(cluster)

        return result


    @staticmethod
    def __item_to_cluster(idx: int, item: object) -> Cluster | None:
        if not isinstance(item, dict):
            logger.warning("Clustering: skipping non-dict cluster item at index %d", idx)
            return None
        name = item.get("name") or f"Cluster {idx + 1}"
        description = item.get("description") or ""

        return Cluster(
            id=f"C{idx + 1:02d}",
            name=str(name),
            description=str(description),
        )


    @staticmethod
    def __build_user_content(questions: List[str]) -> str:
        lines = [f"[{i}] {q}" for i, q in enumerate(questions)]
        return CLUSTER_USER_PROMPT_TEMPLATE.format(questions_text="\n".join(lines))
