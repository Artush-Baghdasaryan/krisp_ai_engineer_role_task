from typing import Optional

from pydantic import BaseModel, Field


class Question(BaseModel):
    id: int = Field(..., description="Unique identifier of the question (e.g. row index).")
    text: str = Field(..., description="Raw question text from the customer.")
    label: Optional[str] = Field(
        default=None,
        description="Ground-truth label (intent/category) from the dataset; used only for evaluation.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Optional embedding vector for the question text.",
    )
