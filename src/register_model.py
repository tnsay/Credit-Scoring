from mlflow.tracking import MlflowClient

MODEL_NAME = "RandomForest_model"
ALIAS_NAME = "staging"
TARGET_VERSION = "2"

client = MlflowClient()

# Remove alias from version 3
try:
    alias_map = client.get_model_version_by_alias(MODEL_NAME, ALIAS_NAME)
    if alias_map.version == "3":
        print(f"❌ Removing alias '{ALIAS_NAME}' from version 3")
        client.delete_registered_model_alias(name=MODEL_NAME, alias=ALIAS_NAME)
except Exception as e:
    print(f"ℹ️ Alias '{ALIAS_NAME}' not found or already clean: {e}")

# Assign version 2 to staging
try:
    print(f"✅ Promoting version {TARGET_VERSION} to alias '{ALIAS_NAME}'")
    client.set_registered_model_alias(
        name=MODEL_NAME, alias=ALIAS_NAME, version=TARGET_VERSION
    )
    print(f"🎉 Version {TARGET_VERSION} is now aliased as '{ALIAS_NAME}'")
except Exception as e:
    print(f"❌ Failed to assign alias to version {TARGET_VERSION}: {e}")

# Promote version 2 to actual stage "Staging" (used by pyfunc.load_model)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=2,
    stage="Staging",  # must match MLflow expected stage
    archive_existing_versions=True,
)
