from matplotlib import pyplot as plt
import mlrose_hiive
import time
import numpy as np

fitness_fn = mlrose_hiive.Queens()
problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness_fn, maximize=False, max_val=8)

pop_sizes = [200, 400, 600, 800, 1000] 
keep_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]

results = {}
for pop_size in pop_sizes:
    for keep_pct in keep_pcts:
        start_time = time.time()
        _, best_fitness, _ = mlrose_hiive.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, random_state=1)
        end_time = time.time()
        
        results[(pop_size, keep_pct)] = (best_fitness, end_time - start_time)




fitness_values = np.array([[results[(pop, pct)][0] for pct in keep_pcts] for pop in pop_sizes])
plt.figure(figsize=(10, 6))
for i, pop_size in enumerate(pop_sizes):
    plt.plot(keep_pcts, fitness_values[i, :], label=f'Pop size: {pop_size}')
plt.title('Fitness Improvement with Varying MIMIC Parameters - 8-Queens')
plt.xlabel('Keep Percentage')
plt.ylabel('Best Fitness Achieved')
plt.legend()
plt.grid(True)
plt.show()

time_values = np.array([[results[(pop, pct)][1] for pct in keep_pcts] for pop in pop_sizes])
plt.figure(figsize=(10, 6))
for i, pop_size in enumerate(pop_sizes):
    plt.plot(keep_pcts, time_values[i, :], label=f'Pop size: {pop_size}')
plt.title('Computational Time with Varying MIMIC Parameters on the 8-Queens Problem')
plt.xlabel('Keep Percentage')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()