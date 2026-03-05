import logging
from typing import Set

import faiss
import numpy as np
import pandas as pd

from src.core import Config
from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DedupeService:
    def __init__(self, embedding_service: EmbeddingService, config: Config):
        self._embedding_service = embedding_service
        self._question_column = config.question_column
        self._normalized_column = "_normalized_questions"
        self._topk = 15
        self._similarity_threshold = 0.80


    def exact_dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self._normalized_column] = self.__normalize_questions(df[self._question_column])

        before = len(df)
        df = df.drop_duplicates(subset=[self._normalized_column])
        df = df.drop(columns=[self._normalized_column])

        logger.info("exact_dedupe: %d -> %d rows", before, len(df))
        return df


    async def semantic_dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        self.__validate_question_column(df)

        texts = df[self._question_column].astype(str).tolist()
        embeddings = await self._embedding_service.embed(texts)
        X = self.__embeddings_to_matrix(embeddings)

        index = self.__build_faiss_index(X)
        dropped = self.__find_duplicate_indices(X, index)

        keep_mask = np.ones(len(df), dtype=bool)
        keep_mask[list(dropped)] = False
        out = df.loc[keep_mask].copy()

        logger.info("semantic_dedupe: %d -> %d rows", len(df), len(out))
        return out.reset_index(drop=True)


    @staticmethod
    def __normalize_questions(series: pd.Series) -> pd.Series:
        return series.astype(str).str.lower().str.strip()


    def __validate_question_column(self, df: pd.DataFrame) -> None:
        if self._question_column not in df.columns:
            raise ValueError(
                f"Column '{self._question_column}' not found in dataframe."
            )


    @staticmethod
    def __embeddings_to_matrix(embeddings: list) -> np.ndarray:
        X = np.asarray(embeddings, dtype="float32")
        if X.ndim != 2:
            raise ValueError("Embeddings must form a 2D matrix (N, D).")
        return X


    @staticmethod
    def __build_faiss_index(x: np.ndarray) -> faiss.Index:
        faiss.normalize_L2(x)
        dim = x.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(x)
        return index


    def __find_duplicate_indices(
        self, x: np.ndarray, index: faiss.Index
    ) -> Set[int]:
        dropped: Set[int] = set()
        n = len(x)
        for i in range(n):
            if i in dropped:
                continue
            sims, idxs = index.search(x[i : i + 1], self._topk)
            sims = sims[0]
            idxs = idxs[0]
            for sim, j in zip(sims, idxs):
                if j == -1 or j == i:
                    continue
                if sim >= self._similarity_threshold:
                    dropped.add(int(j))
        return dropped
