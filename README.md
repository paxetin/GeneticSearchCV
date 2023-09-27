# GeneticSearchCV: Genetic Algorithm-based Hyperparameter Optimization

This repository provides a GeneticSearchCV class that utilizes a Genetic Algorithm (GA) for hyperparameter optimization of machine learning models. 
The code is designed for both single and multi-objective optimization problems.
This methods finds the best parameter faster than grid search.

This class now supports both classification model and regression model from sci-kit learn packages. 
I will progressively update more methods for other type of task.

## GeneticSearchCV Class

- The core class for GA-based hyperparameter optimization.
- The default genetic algorithm is NSGA-II (Non-dominated Sorting Genetic Algorithm II).
- Supports both single and multi-objective optimization tasks.
- Allows customization of Pymoo evolutionary algorithm with specified parameters.
- Utilizes a custom ParamProblem instance for optimization.
- For the parameter `scoring`, please refer to: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

## Installation

To install the required packages for this code, follow these steps:

1. Clone this GitHub repository.

    ```bash
    git clone https://github.com/paxetin/GeneticSearchCV.git
    ```

2. Create a virtual environment (recommended) using `virtualenv` or `conda`.

3. Activate the virtual environment:

   ```bash
   source <virtual_env_name>/bin/activate  # On Linux/Mac
   ```

   ```bash
   <virtual_env_name>\Scripts\activate  # On Windows
   ```

4. Navigate to the repository directory and run:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the requirements, you can use the GeneticSearchCV class for hyperparameter optimization in your Python projects.

```python
from evolutionary_search import GeneticSearchCV

# Define your machine learning model and hyperparameter grid
estimator = YourEstimatorClass()
param_grid = {
    'param1': [value1, value2],
    'param2': [value3, value4],
    # Add more hyperparameters here
}

# Create a GeneticSearchCV instance and fit it to your data
genetic_search = GeneticSearchCV(estimator, param_grid)
genetic_search.fit(X, y)

# Access the best model and hyperparameters
best_model = genetic_search.best_estimator_
best_params = genetic_search.best_params_
best_score = genetic_search.best_score_
```

This code provides a flexible framework for optimizing hyperparameters using genetic algorithms, helping you enhance the performance of your machine learning models.