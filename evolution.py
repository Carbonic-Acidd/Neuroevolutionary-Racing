import numpy as np

from brain import Brain
from car import Car
from copy import deepcopy,copy
import random

from constants import ATTRIBUTE_NAMES, MUTATION_RATE, MUTATION_STRENGTH, N_SURVIVORS

def crossover(brain_a, brain_b):
    child_brain = Brain()
    for attribute_name in ATTRIBUTE_NAMES:
        a_values, b_values = getattr(brain_a, attribute_name), getattr(brain_b, attribute_name)
        take_from_a = np.random.rand(*a_values.shape) < 0.5
        setattr(child_brain, attribute_name, np.where(take_from_a, a_values, b_values))
    return child_brain

def mutate(brain, rate = MUTATION_RATE, strength = MUTATION_STRENGTH):
    for attribute_name in ATTRIBUTE_NAMES:
        values = getattr(brain,attribute_name)
        should_mutate = np.random.rand(*values.shape) < rate
        noise = np.random.uniform(-strength, strength, size= values.shape)
        setattr(brain, attribute_name, values + should_mutate * noise)
    return brain

def next_generation(cars, start_pos, start_heading):

    ranked = sorted(cars, key = lambda car: car.fitness(), reverse = True)
    survivors = ranked[:N_SURVIVORS]

    new_gen = []
    #Elitism : Keep top Car
    new_gen.append(Car(start_pos,heading = start_heading, brain= deepcopy(survivors[0].brain)))

    #Immigrant: Completely Random Brain
    new_gen.append(Car(start_pos, heading = start_heading))

    #Crossover and Mutate the rest of the necessary cars
    while len(new_gen) < len(cars):
        parent_a, parent_b = random.sample(survivors, 2)
        child_brain = mutate(crossover(parent_a.brain, parent_b.brain))
        new_gen.append(Car(start_pos,heading = start_heading, brain = child_brain))

    return new_gen