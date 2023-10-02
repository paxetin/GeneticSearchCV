from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from problems import soo, moo
from typing import *
import json
import numpy as np

# Define a genetic search optimization class
class GeneticSearchCV:
    """
    Genetic Algorithm-based Hyperparameter Optimization.

    This class provides a genetic algorithm-based approach for hyperparameter optimization
    of machine learning models. It supports both single and multi-objective optimization tasks.

    Parameters:
        estimator (object): The machine learning model to optimize.
        param_grid (dict): A dictionary of hyperparameter grids to be searched.
        scoring (str or list, optional): Scoring metric(s) to optimize.
        cv (int, optional): Number of cross-validation folds.
        algorithm (object, optional): Evolutionary algorithm from Pymoo packages.
        ga_params (dict, optional): Genetic algorithm parameters.

    Methods:
        fit(x, y): Fit the GeneticSearchCV instance to the data.

    Attributes:
        best_estimator_ (object): The best machine learning model found.
        best_score_ (float or list): The best score(s) achieved.
        best_params_ (dict): The best hyperparameter combination found.
    """
    def __init__(self, 
                 estimator: object,
                 param_grid: dict, 
                 scoring: Optional[str or list]=None,
                 cv: Optional[int]=None,
                 algorithm: Optional[object]=None,
                 ga_params: Optional[dict]=None,
                 encode_type: str='int'):
        
        # Initialize the genetic search with the given parameters
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.encode_type = encode_type

        # Determine whether it's a single-objective optimization problem or multi-objective optimization problem
        if type(scoring) == list and len(scoring) > 1:
            self.MOO = True
            self.scoring = scoring
            self.problem = moo.MOO(self.estimator, self.param_grid, self.scoring, self.cv, self.encode_type)
        else:
            self.MOO = False
            self.scoring = "".join(scoring) if type(scoring) == list else scoring
            self.problem = soo.SOO(self.estimator, self.param_grid, self.scoring, self.cv, self.encode_type)

        # Set genetic algorithm parameters or use default values
        self.ga_params = ga_params if ga_params else self._default_ga_params()
        for key, value in self.ga_params.items():
            setattr(self, key, value)
        
        # Create the NSGA2 algorithm instance if algorithm is not provided
        if algorithm:
            self.algorithm = algorithm
        else:
            if self.encode_type == 'binary':
                self.algorithm = NSGA2(pop_size=self.POPULATIONSIZE,
                                        sampling=BinaryRandomSampling(),
                                        crossover=SinglePointCrossover(prob=self.CX_RATE),
                                        mutation=BitflipMutation(prob=self.MUTATION_RATE),
                                        eliminate_duplicates=True)
            else:
                self.algorithm = NSGA2(pop_size=self.POPULATIONSIZE,
                                        crossover=SBX(prob=self.CX_RATE, eta=200),
                                        mutation=PolynomialMutation(prob=self.MUTATION_RATE),
                                        eliminate_duplicates=True)  

        # Define the termination condition
        self.termination = DefaultMultiObjectiveTermination(n_max_gen=self.NGEN)
        
        # Initialize the attributes for best params
        self.best_estimator_ = None
        self.best_score_ = [0] * len(self.scoring) if self.MOO else 0
        self.best_params_ = None

    # Default genetic algorithm parameters
    def _default_ga_params(self):
        with open('./default_params/ga_params.json', 'r') as p:
            ga_params = json.load(p)
        return ga_params

    # Fit the genetic search to the data
    def fit(self, x, y):
        # Fit data to the problem
        self.problem.set_data(x, y)

        # Running the optimization algorithm and store the result in res
        res = minimize(self.problem,
                       self.algorithm,
                       self.termination,
                       verbose=False)

        # Select the best model from the pareto-front space
        out = res.X.astype(int) if len(res.X.shape) > 1 else res.X.reshape(1, -1).astype(int)
        for individual in out:
            S = self.problem.decode(individual)
            c = ''.join(map(lambda x: str(x), S.values()))
            if self._less_is_better(c):
                self.best_score_ = self.problem.seen_combinations[c][:-1]
                self.best_params_ = S
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)

    def _less_is_better(self, c):
        if self.MOO:
            return np.mean(self.problem.seen_combinations[c][:-1]) < np.mean(self.best_score_)
        else:
            return self.problem.seen_combinations[c][0] < self.best_score_