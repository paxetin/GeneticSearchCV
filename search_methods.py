from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from sklearn.model_selection import cross_validate
import numpy as np
from typing import *


class ParamProblem(Problem):
    """
    Custom Problem class for hyperparameter optimization.

    This class defines a custom optimization problem for hyperparameter optimization.
    It supports both single and multi-objective optimization tasks.

    Parameters:
        estimator (object): The machine learning model to optimize.
        x (array-like): The training data.
        y (array-like): The target labels.
        param_grid (dict): A dictionary of hyperparameter grids to be searched.
        scoring (str or list, optional): Scoring metric(s) to optimize.
        cv (int, optional): Number of cross-validation folds.
    """
    def __init__(self, estimator, x, y, param_grid, scoring, cv):
        # Initialize the problem with the given parameters
        self.estimator = estimator
        self._x = x
        self._y = y
        self.params = param_grid
        self.cv = cv

        # Calculate the binary representation length for each parameter in param_grid
        self.bit_len = list(map(lambda x: len(x).bit_length(), self.params.values()))
        self.seen_combinations = {}

        # Determine if it's a multi-objective optimization problem
        if type(scoring) == list:
            self.MOO = True
            self.scoring = scoring
            self.n_obj = len(self.scoring)
        else:
            self.MOO = False
            self.scoring = scoring if type(scoring) == str else 'score'
            self.n_obj = 1

        self.neg_scores = ['neg_brier_score', 'neg_log_loss']

        # Initialize the problem with the appropriate number of decision variables, objectives, and constraints
        super().__init__(n_var=sum(self.bit_len),
                         n_obj=self.n_obj,
                         n_ieq_constr=1)

    # Custom evaluation method for the optimization problem
    def _evaluate(self, x, out, *args, **kwargs):
        population = x.astype(int)
        Fs, Gs = self._init_results()

        for individual in population:
            S = self.binary_decode(individual)
            fvalues, gvalue = self._get_fitness_values(S)
            Gs.append(gvalue)
            if self.MOO:
                for i, k in enumerate(Fs.keys()):
                    f = -fvalues[i] if self.scoring[i] not in self.neg_scores else fvalues
                    Fs[k].append(f)
            else:
                f = -fvalues if self.scoring[i] not in self.neg_scores else fvalues
                Fs['fvalue1'].append(f)

        out_f = [Fs[f'fvalue{int(i+1)}'] for i in range(len(Fs.keys()))]

        out["F"] = np.column_stack(list(out_f))
        out["G"] = np.column_stack(list(Gs))

    # Decode the binary representation of parameters to their original values
    def binary_decode(self, individual):
        S, loc = {}, 0
        for i, l in enumerate(self.bit_len):
            decoded_idx = int(''.join(map(str, individual[loc:loc+l])), 2)
            if decoded_idx < len(self.params[list(self.params.keys())[i]]): 
                S[list(self.params.keys())[i]] = self.params[list(self.params.keys())[i]][decoded_idx]
                loc += l
            else: 
                return None
        return S

    # Calculate fitness values and constraints for a given parameter combination
    def _get_fitness_values(self, S):
        if S is None:
            if self.MOO:
                fvalues = [0 for _ in range(len(self.scoring))]
            else:
                fvalues = 0
            return (fvalues, 1)
        combination = ''.join(map(lambda x: str(x), S.values()))
        if combination in self.seen_combinations.keys(): 
            [fvalues, gvalue] = self.seen_combinations[combination]
        else:
            fvalues, gvalue = self._get_result(S) 
            self.seen_combinations[combination] = [fvalues, gvalue]
        return (fvalues, gvalue)

    # Calculate fitness values by cross-validation using the estimator
    def _get_result(self, S):
        cls = self.estimator.set_params(**S)
        cv_result = cross_validate(cls, self._x, self._y, cv=self.cv, scoring=self.scoring)
        if self.MOO:
            fvalues = []
            for v in list(cv_result.values())[2:]:
                fvalues += [np.mean(v)]
        else:
            fvalues = np.mean(cv_result[f'test_{self.scoring}'])
        return fvalues, 0

    # Initialize the result structures for objectives and constraints
    def _init_results(self):
        if self.MOO:
            result = {}
            for i in range(len(self.scoring)):
                result[f'fvalue{int(i+1)}'] = []
            return result, []
        else:
            return {'fvalue1': []}, []

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
                 ga_params: Optional[dict]=None):
        
        # Initialize the genetic search with the given parameters
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        
        # Set genetic algorithm parameters or use default values
        self.ga_params = ga_params if ga_params else self._default_ga_params()
        for key, value in self.ga_params.items():
            setattr(self, key, value)
        
        # Create the NSGA2 algorithm instance for multi-objective optimization
        self.algorithm = NSGA2(pop_size=self.POPULATIONSIZE,
                                sampling=BinaryRandomSampling(),
                                crossover=SinglePointCrossover(prob=self.CX_RATE),
                                mutation=BitflipMutation(prob=self.MUTATION_RATE),
                                eliminate_duplicates=True)

        # Define the termination condition
        self.termination = DefaultMultiObjectiveTermination(n_max_gen=self.NGEN)
        
        self.best_estimator_ = None
        self.best_score_ = 0 if type(self.scoring) != list else [0 for _ in range(len(self.scoring))]
        self.best_params_ = None

    # Default genetic algorithm parameters
    def _default_ga_params(self):
        return {
            "POPULATIONSIZE": 25, 
            "NGEN": 100,
            "CX_RATE": 0.9,
            "MUTATION_RATE": 0.1
        }

    # Fit the genetic search to the data
    def fit(self, x, y):
        self.problem = ParamProblem(self.estimator, x, y, self.param_grid, self.scoring, self.cv)
        res = minimize(self.problem,
                       self.algorithm,
                       self.termination,
                       verbose=False)
        
        for individual in res.X.astype(int):
            S = self.problem.binary_decode(individual)
            c = ''.join(map(lambda x: str(x), S.values()))
            if self.problem.seen_combinations[c][0] > self.best_score_:
                self.best_score_ = self.problem.seen_combinations[c][0]
                self.best_params_ = S
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
