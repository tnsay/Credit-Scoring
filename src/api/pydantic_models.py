from pydantic import BaseModel, Field, conint, confloat
import pandas as pd

# Define your constrained types first
TotalAmount = confloat(ge=-1e6, le=1e6)
TransactionCount = conint(ge=0)
TransactionHour = conint(ge=0, le=23)
TransactionDay = conint(ge=1, le=31)
TransactionMonth = conint(ge=1, le=12)
TransactionYear = conint(ge=2000, le=2030)


# data validation
class CustomerData(BaseModel):
    total_amount: float = Field(...,
                                ge=-1e6,
                                le=1e6,
                                example=-10000.0,
                                description="Total transaction amount")
    avg_amount: float = Field(
        ..., example=-5000.0, description="Average transaction amount"
    )
    std_amount: float = Field(
        ..., example=0.0, description="Standard deviation of amounts"
    )
    transaction_count: int = Field(
        ..., ge=0, example=1, description="Number of transactions"
    )
    transaction_hour: int = Field(
        ..., ge=0, le=23, example=16, description="Hour of transaction"
    )
    transaction_day: int = Field(
        ..., ge=1, le=31, example=21, description="Day of month of transaction"
    )
    transaction_month: int = Field(
        ..., ge=1, le=12, example=11, description="Month of transaction"
    )
    transaction_year: int = Field(
        ..., ge=2000, le=2030, example=2018, description="Year of transaction"
    )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.dict()])


class PredictionResponse(BaseModel):
    risk_score: float
