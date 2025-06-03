import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Cargar el último modelo entrenado
mlflow.set_tracking_uri("file:///app/mlruns")

try:
    experiment = mlflow.get_experiment_by_name("Iris-Classification")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    last_run_id = runs.iloc[0].run_id
    model = mlflow.pyfunc.load_model(f"runs:/{last_run_id}/model")
    print(f"Model loaded from run: {last_run_id}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# ✅ Definir un esquema de entrada para FastAPI
class Features(BaseModel):
    values: list[float]

@app.post("/predict")
def predict(features: Features):
    try:
        input_data = np.array(features.values).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
