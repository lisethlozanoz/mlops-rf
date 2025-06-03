import requests
import json

url = "http://localhost:1234/predict"

data = {
    "features": [5.1, 3.5, 1.4, 0.2]  # Ejemplo de datos de entrada
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)