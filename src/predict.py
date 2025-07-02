import pandas as pd
import mlflow
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load input data
def load_input(input_path: str) -> pd.DataFrame:
    try:
        full_path = os.path.join(ROOT_DIR, input_path)
        data = pd.read_csv(full_path)
        print(f"✅ Loaded input data from {full_path}")
        return data
    except Exception as e:
        print(f"❌ Failed to load input data: {e}")
        sys.exit(1)

# Load model from the registry
def load_model(model_name: str, stage: str = "Staging"):
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Loaded model from Registry: {model_uri}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

# Run inference
def predict(model, input_df: pd.DataFrame):
    try:
        # Save CustomerId for output if it exists
        customer_ids = input_df["CustomerId"] if "CustomerId" in input_df.columns else None
        
        # Drop columns not used in training
        input_features = input_df.drop(columns=["CustomerId", "is_high_risk"], errors="ignore")
        
        # Predict
        preds = model.predict(input_features)
        
        # Return DataFrame with CustomerId and prediction
        if customer_ids is not None:
            return pd.DataFrame({
                "CustomerId": customer_ids,
                "prediction": preds
            })
        else:
            return pd.DataFrame({"prediction": preds})
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)

# Save predictions
def save_output(pred_df, output_path: str):
    try:
        pred_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")
        sys.exit(1)

# Main logic
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using MLflow model registry")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model-name", type=str, default="RandomForest_model", help="Name of the registered model")
    parser.add_argument("--stage", type=str, default="Staging", help="Model registry stage (e.g., Staging or Production)")

    args = parser.parse_args()

    # Pipeline
    input_df = load_input(args.input)
    model = load_model(args.model_name, args.stage)
    pred_df = predict(model, input_df)
    save_output(pred_df, args.output)
