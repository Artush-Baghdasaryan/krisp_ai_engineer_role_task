from typing import Iterator, List

import pandas as pd

from src.core import Config
from src.models import Question

class DataLoader:
    def __init__(self, config: Config):
        self.dataset_path = config.dataset_path
        self.question_col = config.question_column
        self.batch_size = config.batch_size

    
    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)

        if self.question_col not in df.columns:
            raise ValueError(f"Question column {self.question_col} not found in dataset")

        return df.copy()
    

    def stream_dataframe(self) -> Iterator[pd.DataFrame]:
        columns = [self.question_col]
        for chunk in pd.read_csv(self.dataset_path, usecols=columns, chunksize=self.batch_size):
            chunk = chunk.reset_index(drop=True)
            yield chunk


    def to_questions(self, df: pd.DataFrame) -> List[Question]:
        return [
            Question(id=idx, text=str(row[self.question_col]))
            for idx, (_, row) in enumerate(df.iterrows())
        ]
        

