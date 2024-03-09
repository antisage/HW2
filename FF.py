import mlrose_hiive
import time
import numpy as np

# Define the Flip Flop problem
problem_size = 100
fitness_fn = mlrose_hiive.FlipFlop()
problem = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness_fn, maximize=True)

# Function to run an experiment
def run_experiment(algorithm, problem, **kwargs):
    start_time = time.time()
    best_state, best_fitness, _ = algorithm(problem, **kwargs)
    end_time = time.time()
    return best_fitness, end_time - start_time

# Algorithm configurations
algorithms = {
    "RHC": {"algorithm": mlrose_hiive.random_hill_climb, "tuned_params": {"restarts": 548}, "default_params": {}},
    "SA": {"algorithm": mlrose_hiive.simulated_annealing, "tuned_params": {"schedule": mlrose_hiive.ExpDecay(init_temp=30, exp_const=0.1, min_temp=0.001)}, "default_params": {}}
    #"GA": {"algorithm": mlrose_hiive.genetic_alg, "tuned_params": {"pop_size": 250, "mutation_prob": 0.22}, "default_params": {}}
    #"MIMIC": {"algorithm": mlrose_hiive.mimic, "tuned_params": {"pop_size": 500, "keep_pct": 0.2}, "default_params": {}}
}

results = {}

# Run experiments for each algorithm with both default and tuned parameters
for name, config in algorithms.items():
    default_fitness, default_time = run_experiment(config["algorithm"], problem, **config["default_params"], random_state=1, max_attempts=10, max_iters=np.inf)
    tuned_fitness, tuned_time = run_experiment(config["algorithm"], problem, **config["tuned_params"], random_state=1, max_attempts=10, max_iters=np.inf)
    results[name] = {"Default Fitness": default_fitness, "Default Time": default_time, "Tuned Fitness": tuned_fitness, "Tuned Time": tuned_time}

# Print the results
for algo, metrics in results.items():
    print(f"{algo}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print("\n")