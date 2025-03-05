class Neuron:
    def __init__(self, j):
        # Constructor to initialize weights based on the provided array
        self.activation = 0
        self.weightv = [j[i] for i in range(4)]

    def act(self, m, x):
        # Activation function to calculate the weighted sum
        a = sum(x[i] * self.weightv[i] for i in range(m))
        return a


class Network:
    def __init__(self, a, b, c, d):
        # Constructor to create a network with four neurons
        self.nrn = [Neuron(a), Neuron(b), Neuron(c), Neuron(d)]
        self.output = [0, 0, 0, 0]

    def threshld(self, k):
        # Threshold function
        if k >= 0:
            return 1
        else:
            return 0

    def activation(self, patrn):
        # Activation function for the entire network
        for i in range(4):
            print(f"\nnrn[{i}].weightv is {self.nrn[i].weightv}")
            self.nrn[i].activation = self.nrn[i].act(4, patrn)
            print(f"Activation is {self.nrn[i].activation}")
            self.output[i] = self.threshld(self.nrn[i].activation)
            print(f"Output value is {self.output[i]}\n")


def main():
    patrn1 = [1, 0, 1, 0]
    wt1 = [0, -3, 3, -3]
    wt2 = [-3, 0, -3, 3]
    wt3 = [3, -3, 0, -3]
    wt4 = [-3, 3, -3, 0]

    print("THIS PROGRAM IS FOR A HOPFIELD NETWORK WITH A SINGLE LAYER OF")
    print("4 FULLY INTERCONNECTED NEURONS. THE NETWORK SHOULD RECALL THE")
    print("PATTERNS 1010 AND 0101 CORRECTLY.\n")

    # Create the network by calling its constructor.
    # The constructor calls the neuron constructor as many times as the number of
    # neurons in the network.
    h1 = Network(wt1, wt2, wt3, wt4)

    # Present a pattern to the network and get the activations of the neurons
    h1.activation(patrn1)

    # Check if the pattern given is correctly recalled and give a message
    for i in range(4):
        if h1.output[i] == patrn1[i]:
            print(f"\nPattern = {patrn1[i]}, Output = {h1.output[i]}, Component matches")
        else:
            print(f"\nPattern = {patrn1[i]}, Output = {h1.output[i]}, Discrepancy occurred")
    print("\n\n")

    patrn2 = [0, 1, 0, 1]
    h1.activation(patrn2)

    for i in range(4):
        if h1.output[i] == patrn2[i]:
            print(f"\nPattern = {patrn2[i]}, Output = {h1.output[i]}, Component matches")
        else:
            print(f"\nPattern = {patrn2[i]}, Output = {h1.output[i]}, Discrepancy occurred")


if __name__ == "__main__":
    main()
