import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlrose_hiive
import numpy as np
from sklearn.metrics import accuracy_score
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current Working Directory: ", os.getcwd())

# Load dataset
df = pd.read_csv('adult_tr.csv')

# Assume the target variable is 'quality' and all others are features
X = df.drop('threshold', axis=1)
y = df['threshold']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))


# Preprocess the dataset
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Apply transformation
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


def train_evaluate_nn(algorithm, parameter_value, X_train, y_train, X_test, y_test):
    start_time = time.time()  # Start time
    if algorithm == 'random_hill_climb':
        print('started RHC')
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[10, 5], activation='relu',
                                              algorithm=algorithm, max_iters=1000, restarts=parameter_value,
                                              bias=True, is_classifier=True, learning_rate=0.1,
                                              early_stopping=True, clip_max=5, max_attempts=100,
                                              random_state=42)
    elif algorithm == 'simulated_annealing':
        print('started SA')
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[10, 5], activation='relu',
                                              algorithm=algorithm, schedule=mlrose_hiive.GeomDecay(init_temp=parameter_value),
                                              max_iters=1000, bias=True, is_classifier=True, learning_rate=0.1,
                                              early_stopping=True, clip_max=5, max_attempts=100,
                                              random_state=42)
    elif algorithm == 'genetic_alg':
        print('started GA')
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[10, 5], activation='relu',
                                              algorithm=algorithm, pop_size=parameter_value,
                                              max_iters=1000, bias=True, is_classifier=True, learning_rate=0.1,
                                              early_stopping=True, clip_max=5, max_attempts=100,
                                              random_state=42)
    else:
        return None

    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    end_time = time.time() 
    computation_time = end_time - start_time 
    return test_accuracy, computation_time


# Parameter ranges
rhc_max_iters = [200, 400, 600, 800, 1000]
sa_init_temps = [1, 10, 50, 100, 500]
ga_pop_sizes = [100, 200, 300, 400, 500]

# Storage for results
rhc_accuracies = []
sa_accuracies = []
ga_accuracies = []

results = {
    'RHC': {'accuracy': [], 'time': []},
    'SA': {'accuracy': [], 'time': []},
    'GA': {'accuracy': [], 'time': []}
}

parameter_values = {
    'RHC': [100, 200, 300, 400, 500], 
    'SA': [1, 10, 50, 100, 500],  
    'GA': [100, 200, 300, 400, 500] 
}

# Run experiments
for iters in parameter_values['RHC']:
    accuracy, time_taken = train_evaluate_nn('random_hill_climb', iters, X_train, y_train, X_test, y_test)
    results['RHC']['accuracy'].append(accuracy)
    results['RHC']['time'].append(time_taken)

for temp in sa_init_temps:
    accuracy, time_taken = train_evaluate_nn('simulated_annealing', temp, X_train, y_train, X_test, y_test)
    results['SA']['accuracy'].append(accuracy)
    results['SA']['time'].append(time_taken)

for pop_size in ga_pop_sizes:
    accuracy, time_taken = train_evaluate_nn('genetic_alg', pop_size, X_train, y_train, X_test, y_test)
    results['GA']['accuracy'].append(accuracy)
    results['GA']['time'].append(time_taken)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plotting accuracy
for alg in results:
    axs[0].plot(parameter_values[alg], results[alg]['accuracy'], label=alg, marker='o')
axs[0].set_title('Test Accuracy by Algorithm')
axs[0].set_xlabel('Parameter Value')
axs[0].set_ylabel('Test Accuracy')
axs[0].legend()

# Plotting computation time
for alg in results:
    axs[1].plot(parameter_values[alg], results[alg]['time'], label=alg, marker='o')
axs[1].set_title('Computation Time by Algorithm')
axs[1].set_xlabel('Parameter Value')
axs[1].set_ylabel('Time (seconds)')
axs[1].legend()

plt.tight_layout()
plt.show()