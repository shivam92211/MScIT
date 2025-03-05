from minisom import MiniSom
import matplotlib.pyplot as plt

data = [
    [0.80, 0.55, 0.22, 0.03],
    [0.82, 0.50, 0.23, 0.03],
    [0.80, 0.54, 0.22, 0.03],
    [0.80, 0.53, 0.26, 0.03],
    [0.79, 0.56, 0.22, 0.03],
    [0.75, 0.60, 0.25, 0.03],
    [0.77, 0.59, 0.22, 0.03]
]

som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5)  # initialization of 6x6 SOM
som.train_random(data, 100)  # trains the SOM with 100 iterations

plt.imshow(som.distance_map())
plt.show()
