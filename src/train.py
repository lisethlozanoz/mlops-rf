import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

def train_model(n_estimators=100, max_depth=5):
    # Configurar MLflow
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("Iris-Classification")
    
    # Cargar datos
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    with mlflow.start_run() as run:
        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Loggear parámetros y métricas
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metric("accuracy", accuracy)
        
        # Guardar modelo
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    
    run_id = train_model(args.n_estimators, args.max_depth)
    with open("src/run_id.txt", "w") as f:
        f.write(run_id)