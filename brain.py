from constants import N_HIDDEN, N_INPUTS, N_OUTPUTS
from network import random_biases, random_weights


class Brain:
    def __init__(self):
        self.hidden_weights = random_weights(N_INPUTS, N_HIDDEN)
        self.hidden_biases = random_biases(N_HIDDEN)
        self.output_weights = random_weights(N_HIDDEN, N_OUTPUTS)
        self.output_biases = random_biases(N_OUTPUTS)
