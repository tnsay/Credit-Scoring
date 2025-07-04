import mlflow

# Set tracking URI if needed (optional if using default local `mlruns/`)
mlflow.set_tracking_uri("file:./mlruns")

# Define the model URI in the registry
model_uri = "models:/RandomForest_model/2"

# Download model files to local folder
downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

print(f"Model exported to: {downloaded_path}")
