import matplotlib.pyplot as plt
import requests

from uncertainty_ellipsoid.plots import visualize

# run make api in the terminal to start the server

URL = "http://localhost:8000"

# Example prediction request
data = {
    "pixel_coordinates": [240, 320],
    "depth": 0.45,
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
        -0.5,
        1.2,
        1.5,
        -0.1,
        0.1,
        -0.1,
        0.1,
        -0.15,
        -0.1,
    ],
    "num_samples": 100,
}

# Check the API health
response = requests.get(URL + "/health")
print(response.json())


# Make prediction request
response = requests.post(URL + "/predict", json=data)
result = response.json()
print(result)
centers = result["center"]
L_elements = result["L_matrix"]

# Example simulation request
response = requests.post(URL + "/simulate", json=data)
result = response.json()
# print(result)
world_coords = result["world_coords"]

## Visualize the results
fig, ax = plt.figure(), plt.gca()
visualize(fig, world_coords, centers, L_elements)
