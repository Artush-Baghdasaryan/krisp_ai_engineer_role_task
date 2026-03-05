import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    llm_model = os.getenv("LLM_MODEL")

    batch_size = int(os.getenv("BATCH_SIZE"))

    dataset_path = os.getenv("DATASET_PATH")
    question_column = os.getenv("QUESTION_COLUMN")
    label_column = os.getenv("LABEL_COLUMN")

    def validate(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

