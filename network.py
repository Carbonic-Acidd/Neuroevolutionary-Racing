"""
network.py - the car's brain.

A tiny feed-forward network:  5 sensor inputs -> 3 hidden neurons -> 2 outputs

Every neuron does the same thing:
    output = tanh(w1*x1 + w2*x2 + ... + bias)
tanh squashes any number into the range -1..1, which suits steering and throttle.

numpy computes a whole layer at once: (weights @ inputs) is every neuron's
weighted sum in one go. It's exactly the same maths as looping over neurons
(run this file directly to see both give the same answer).
"""
import math

import numpy as np

N_INPUTS = 5    # one per sensor
N_HIDDEN = 3
N_OUTPUTS = 2   # steering, throttle

def random_weights(input_neurons, output_neurons):
    return np.random.uniform(-1,1,size=(output_neurons, input_neurons))

def random_biases(output_neurons):
    return np.random.uniform(-1,1,size=(output_neurons))

def layer(inputs, weights, biases):
    return np.tanh(weights @ inputs + biases)

def forward_pass(inputArray, brain):

    hiddenLayer = layer(inputArray, brain.hidden_weights, brain.hidden_biases)

    outputLayer = layer(hiddenLayer, brain.output_weights, brain.output_biases)

    return outputLayer


