# MLOps Random Forest Iris Classification

Proyecto de ejemplo que entrena un modelo Random Forest para clasificar flores Iris usando MLflow para el tracking, Docker para contenerización y FastAPI para servir el modelo.

## Estructura

- `src/train.py`: Entrena el modelo y lo registra con MLflow.
- `src/serve.py`: API con FastAPI para predecir usando el modelo entrenado.
- `Dockerfile`: Contenedor con Python y dependencias.
- `docker-compose.yml`: Orquesta servicios de entrenamiento y servidor.

## Cómo usar

1. Construir y levantar los contenedores:
   ```bash
   docker compose up --build
   ```
2. Entrenar el modelo (se ejecuta automáticamente con el servicio trainer).
3. Iniciar el servidor FastAPI (servicio server) en [http://localhost:1234](http://localhost:1234).
4. Hacer predicciones enviando un POST a `/predict` con un JSON que contenga las características:
   ```bash
   curl -X POST "http://localhost:1234/predict" -H "Content-Type: application/json" -d "[5.1, 3.5, 1.4, 0.2]"
   ```
   Respuesta esperada:
   ```json
   {"prediction": 0}
   ```

## Notas

- El experimento se guarda localmente en la carpeta `mlruns`.
- Asegúrate de tener Docker y Docker Compose instalados.
- Para desarrollo local sin contenedor, instala dependencias con:
  ```bash
  pip install -r requirements.txt
  ```

## Dependencias Principales

- Python 3.10
- scikit-learn
- mlflow
- fastapi
- uvicorn
- python-multipart

## Interpretación de la predicción

El modelo clasifica las flores Iris en tres clases, representadas por números enteros:

- `0`: Iris Setosa
- `1`: Iris Versicolor
- `2`: Iris Virginica
