import asyncio
from typing import List
from src.core import Config
from openai import AsyncOpenAI

class EmbeddingService:
    def __init__(self, config: Config):
        self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        self.embedding_model = config.embedding_model
        self.concurrency = 4
    

    async def embed(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        semaphore = asyncio.Semaphore(self.concurrency)
        results: List[List[float] | None] = [None] * len(texts)

        async def worker(start: int, text_chunk: List[str]):
            async with semaphore:
                embeddings = await self.__embed_batch(text_chunk)
                for idx, embedding in enumerate(embeddings):
                    results[start + idx] = embedding

        tasks = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tasks.append(asyncio.create_task(worker(i, chunk)))

        await asyncio.gather(*tasks)
        return results


    async def __embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.openai_client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in response.data]

        