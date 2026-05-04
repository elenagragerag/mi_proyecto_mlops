import mlflow
import pickle

print("Conectando a MLFlow...")
mlflow.set_tracking_uri("http://localhost:5001")
client = mlflow.tracking.MlflowClient()

print("Buscando experimentos...")
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experimento: {exp.name}, ID: {exp.experiment_id}")

print("Buscando runs...")
runs = client.search_runs("1", order_by=["metrics.roc_auc DESC"], max_results=1)
print(f"Runs encontrados: {len(runs)}")
best_run_id = runs[0].info.run_id
print("Mejor run:", best_run_id)

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modelo guardado en models/best_model.pkl")
