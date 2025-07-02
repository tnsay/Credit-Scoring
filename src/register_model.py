# import mlflow
# from mlflow.tracking import MlflowClient

# MODEL_NAME = "RandomForest_model"
# ALIAS_NAME = "staging"  # new approach replacing STAGE concept

# client = MlflowClient()
# model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# best_run = None
# best_auc = -1.0

# for mv in model_versions:
#     if mv.version in ("2", "3"):  # skip known problematic versions
#         print(f"⚠️ Skipping version {mv.version} due to serialization issues")
#         continue
#     run_id = mv.run_id
#     try:
#         metrics = client.get_run(run_id).data.metrics
#         roc_auc = float(metrics.get("roc_auc", -1.0))
#         if roc_auc > best_auc:
#             best_auc = roc_auc
#             best_run = mv
#     except Exception as e:
#         print(f"⚠️ Skipping version {mv.version} due to error: {e}")
#         continue

# if best_run:
#     version = best_run.version
#     print(f"Promoting version {version} of {MODEL_NAME} to alias '{ALIAS_NAME}'")
#     try:
#         client.set_registered_model_alias(
#             name=MODEL_NAME,
#             alias=ALIAS_NAME,
#             version=version
#         )
#         print(f"✅ Model version {version} assigned to alias '{ALIAS_NAME}'")
#     except Exception as e:
#         print(f"❌ Failed to promote version {version} to alias '{ALIAS_NAME}': {e}")
# else:
#     print("❌ No valid model version found to promote.")


import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "RandomForest_model"
ALIAS_NAME = "staging"

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

best_run = None
best_auc = -1.0

for mv in model_versions:
    version = mv.version
    run_id = mv.run_id

    # Skip version 3 which causes serialization issues
    if version == "3":
        print(f"⚠️ Skipping version {version} due to known serialization issues")
        continue

    try:
        metrics = client.get_run(run_id).data.metrics
        roc_auc = float(metrics.get("roc_auc", -1.0))
        print(f"ℹ️ Version {version} has ROC AUC: {roc_auc}")
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_run = mv
    except Exception as e:
        print(f"⚠️ Skipping version {version} due to error: {e}")
        continue

# Remove alias from version 3 if it's assigned
try:
    alias_map = client.get_model_version_by_alias(MODEL_NAME, ALIAS_NAME)
    if alias_map.version == "3":
        print(f"❌ Removing alias '{ALIAS_NAME}' from version 3 due to issues")
        client.delete_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS_NAME
        )
except Exception:
    # alias might not exist or version is already clean
    pass

# Assign best valid version to alias
if best_run:
    try:
        version = best_run.version
        print(f"✅ Promoting version {version} to alias '{ALIAS_NAME}'")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS_NAME,
            version=version
        )
        print(f"🎉 Model version {version} is now aliased as '{ALIAS_NAME}'")
    except Exception as e:
        print(f"❌ Failed to assign alias: {e}")
else:
    print("❌ No valid model version found to promote.")
