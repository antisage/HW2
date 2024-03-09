import mlrose_hiive
import time
import matplotlib.pyplot as plt

# Define the 8-Queens problem
fitness_fn = mlrose_hiive.Queens()
problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness_fn, maximize=False, max_val=8)

# Algorithm configurations
algorithms = {
    "RHC": {"func": mlrose_hiive.random_hill_climb, "params": {"restarts": 110}, "default_params": {}},
    "SA": {"func": mlrose_hiive.simulated_annealing, "params": {"schedule": mlrose_hiive.GeomDecay(init_temp=1000, decay=0.90)}, "default_params": {}},
    "GA": {"func": mlrose_hiive.genetic_alg, "params": {"pop_size": 200, "mutation_prob": 0.1}, "default_params": {}},
    "MIMIC": {"func": mlrose_hiive.mimic, "params": {"pop_size": 1000, "keep_pct": 0.20}, "default_params": {}}
}

# Results storage
results = {name: {"tuned": {"fitness": None, "time": None}, "default": {"fitness": None, "time": None}} for name in algorithms}

# Run experiments
for name, alg in algorithms.items():
    for param_type in ["tuned", "default"]:
        params = alg["params"] if param_type == "tuned" else alg["default_params"]
        
        start_time = time.time()
        best_state, best_fitness, _ = alg["func"](problem, **params, random_state=1)
        end_time = time.time()
        
        results[name][param_type]["fitness"] = best_fitness
        results[name][param_type]["time"] = end_time - start_time


# Fitness Scores
labels = list(algorithms.keys())
tuned_fitness = [results[name]["tuned"]["fitness"] for name in labels]
default_fitness = [results[name]["default"]["fitness"] for name in labels]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x, tuned_fitness, width, label='Tuned')
plt.bar([p + width for p in x], default_fitness, width, label='Default Tuning')

plt.ylabel('Best Fitness Achieved')
plt.title('Fitness Scores by Algorithm and Parameter Tuning')
plt.xticks([p + width / 2 for p in x], labels)
plt.legend()
plt.show()

# Computational Times
tuned_times = [results[name]["tuned"]["time"] for name in labels]
default_times = [results[name]["default"]["time"] for name in labels]

plt.figure(figsize=(12, 6))
plt.bar(x, tuned_times, width, label='Tuned')
plt.bar([p + width for p in x], default_times, width, label='Default Tuning')

plt.ylabel('Computation Time (seconds)')
plt.title('Computation Times by Algorithm and Parameter Tuning')
plt.xticks([p + width / 2 for p in x], labels)
plt.legend()
plt.show()