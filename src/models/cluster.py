from pydantic import BaseModel, Field


class Cluster(BaseModel):
    id: str = Field(..., description="Stable short id like 'C01', 'C02' ...")
    name: str = Field(..., description="Short title summarizing the group of questions.")
    description: str = Field(
        ...,
        description="Clear explanation of what kinds of questions belong to this cluster.",
    )
    count: int = Field(
        default=0,
        description="Number of questions assigned to this cluster.",
    )
