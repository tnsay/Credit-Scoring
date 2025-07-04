from mlflow.tracking import MlflowClient

client = MlflowClient()
model_versions = client.search_model_versions("name='RandomForest_model'")
for mv in model_versions:
    print(f"Version {mv.version}, aliases: {mv.aliases}")
