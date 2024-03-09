import numpy as np
import pandas as pd
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import os
import random
import time
import math
import mlrose_hiive

def four_peaks(bitstring, T):
    tail = 0
    head = 0
    n = len(bitstring)
    
    # Count tail 0s
    for i in range(n):
        if bitstring[i] == 0:
            tail += 1
        else:
            break
    
    # Count head 1s
    for i in range(n-1, -1, -1):
        if bitstring[i] == 1:
            head += 1
        else:
            break
    
    if head > T and tail > T:
        return n + head + tail
    else:
        return max(head, tail)
    
def random_bitstring(n):
    return [random.randint(0, 1) for _ in range(n)]

def mutate_bitstring(bitstring):
    mutant = bitstring[:]
    index = random.randint(0, len(bitstring) - 1)
    mutant[index] = 1 - mutant[index]  # Flip the bit
    return mutant

def random_hill_climbing_with_restarts(fitness_func, n, T, max_iterations, restarts):
    best_solution = None
    best_fitness = float('-inf')
    total_evaluations = 0
    start_time = time.time()
    
    for _ in range(restarts + 1):  # +1 to include the initial run before restarts
        current_solution = random_bitstring(n)
        current_fitness = fitness_func(current_solution, T)
        total_evaluations += 1
        
        for _ in range(max_iterations):
            candidate = mutate_bitstring(current_solution)
            candidate_fitness = fitness_func(candidate, T)
            total_evaluations += 1
            
            if candidate_fitness > current_fitness:
                current_solution, current_fitness = candidate, candidate_fitness
                
                if candidate_fitness > best_fitness:
                    best_solution, best_fitness = candidate, candidate_fitness
                    
    end_time = time.time()
    return best_solution, best_fitness, total_evaluations, end_time - start_time

def simulated_annealing(fitness_func, bitstring_length, T, max_iterations, temperature, cooling_rate):
    current_solution = random_bitstring(bitstring_length)
    current_fitness = fitness_func(current_solution, T)
    best_solution = current_solution[:]
    best_fitness = current_fitness
    fitness_over_time = [current_fitness]  # Track fitness over iterations for graphing
    
    for iteration in range(max_iterations):
        temp = temperature * (cooling_rate ** iteration)
        candidate = mutate_bitstring(current_solution)
        candidate_fitness = fitness_func(candidate, T)
        if candidate_fitness > current_fitness or random.random() < math.exp((candidate_fitness - current_fitness) / temp):
            current_solution, current_fitness = candidate, candidate_fitness
            if candidate_fitness > best_fitness:
                best_solution, best_fitness = candidate, candidate_fitness
        fitness_over_time.append(best_fitness)
    
    return best_solution, best_fitness, fitness_over_time


temperatures = [1, 1000, 5000, 10000, 20000]  # Example temperature values
cooling_rates = [0.85, 0.90, 0.95, 0.99]  # Example cooling rate values

problem_size = 1000  # Fixed problem size for all experiments
T = 20  # Threshold for the Four Peaks problem
max_iterations = 1000

# Storage for experiment results
results = {}

for temp in temperatures:
    for cooling_rate in cooling_rates:
        _, best_fitness, _ = simulated_annealing(four_peaks, problem_size, T, max_iterations, temp, cooling_rate)
        results[(temp, cooling_rate)] = best_fitness


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Preparing data for 3D plot
X, Y = np.meshgrid(temperatures, cooling_rates)
Z = np.array([[results[(x,y)] for x in temperatures] for y in cooling_rates])

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Temperature')
ax.set_ylabel('Cooling Rate')
ax.set_zlabel('Best Fitness')
ax.set_title('Effect of Temperature and Cooling Rate on Fitness')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()