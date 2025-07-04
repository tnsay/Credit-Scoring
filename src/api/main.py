from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc

app = FastAPI(
    title="Credit Risk Predictor API",
    description=(
        "API to assess credit risk based on customer "
        "transaction behavior"
    ),
    version="1.0.0",
)

# Load model from MLflow Registry
model_name = "RandomForest_model"
# model_uri = "models:/RandomForest_model/2"


print("🔍 Loading model from: local storage exported_model dir")
model = mlflow.pyfunc.load_model("exported_model")


@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    input_df = data.to_dataframe()
    print(f"📨 Received input: {input_df.to_dict(orient='records')}")
    prediction = model.predict(input_df)
    print(f"🎯 Prediction: {prediction}")
    return PredictionResponse(risk_score=round(prediction[0], 4))
