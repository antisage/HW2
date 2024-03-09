import mlrose_hiive
import matplotlib.pyplot as plt
import time

import numpy as np

# Define the 8-Queens problem
fitness_fn = mlrose_hiive.Queens()
problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness_fn, maximize=False, max_val=8)

restart_values = [0,5,10,25,50,75,100,110,125,150]  # Example restart values
success_rates = []
times_taken = []
average_fitnesses = []


for restarts in restart_values:
    start_time = time.time()
    # Solve the problem using RHC with the current number of restarts
    best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(problem,
                                                              restarts=restarts,
                                                              random_state=1,
                                                              max_attempts=10,  # You may adjust max_attempts as needed
                                                              max_iters=np.inf)
    end_time = time.time()
    
    average_fitnesses.append(best_fitness)
    times_taken.append(end_time - start_time)

initial_temperatures = [1, 10, 100, 1000, 10000]
cooling_rates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,0.95]

computation_times = np.zeros((len(initial_temperatures), len(cooling_rates)))

best_fitnesses = np.zeros((len(initial_temperatures), len(cooling_rates)))

for i, temp in enumerate(initial_temperatures):
    for j, rate in enumerate(cooling_rates):
        _, best_fitness, _ = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(init_temp=temp, decay=rate), max_attempts=10, max_iters=np.inf, random_state=1)
        best_fitnesses[i][j] = best_fitness
        end_time = time.time()
        computation_times[i][j] = end_time - start_time


plt.figure(figsize=(12, 8))
plt.imshow(best_fitnesses, cmap='hot', interpolation='nearest')
plt.colorbar(label='Best Fitness Achieved')
plt.xticks(np.arange(len(cooling_rates)), labels=[str(rate) for rate in cooling_rates], rotation=45)
plt.yticks(np.arange(len(initial_temperatures)), labels=[str(temp) for temp in initial_temperatures])
plt.xlabel('Cooling Rate')
plt.ylabel('Initial Temperature')
plt.title('Tuning Simulated Annealing for the 8-Queens Problem')
plt.show()

plt.figure(figsize=(12, 8))
plt.imshow(computation_times, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Computation Time (seconds)')
plt.xticks(np.arange(len(cooling_rates)), labels=[str(rate) for rate in cooling_rates], rotation=45)
plt.yticks(np.arange(len(initial_temperatures)), labels=[str(temp) for temp in initial_temperatures])
plt.xlabel('Cooling Rate')
plt.ylabel('Initial Temperature')
plt.title('Computation Time for Simulated Annealing on the 8-Queens Problem')
plt.show()