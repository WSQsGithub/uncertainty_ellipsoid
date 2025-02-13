import requests

# Example prediction request
data = {
    "pixel_coordinates": [240, 320],
    "depth": 0.5,
    "uncertainty_set": [
        600,
        610,
        600,
        610,
        310,
        320,
        250,
        260,
        1.2,
        1.5,
        -1.2,
        -0.9,
        1.2,
        1.5,
        -0.1,
        0.1,
        -0.1,
        0.1,
        -0.15,
        -0.1,
    ],
}
# Check the API health
response = requests.get("http://localhost:8000/health")
print(response.json())


# Make prediction request
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(result)
