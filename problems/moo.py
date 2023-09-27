from pymoo.core.problem import Problem
from sklearn.model_selection import cross_validate
import numpy as np

class MOO_cls(Problem):
    """
    Custom Problem class for multi-objectives hyperparameter optimization of classification model.

    This class defines a custom optimization problem for hyperparameter optimization.
    It supports multi-objective optimization tasks.

    Parameters:
        estimator (object): The machine learning model to optimize.
        x (array-like): The training data.
        y (array-like): The target labels.
        param_grid (dict): A dictionary of hyperparameter grids to be searched.
        scoring (str or list, optional): Scoring metric(s) to optimize.
        cv (int, optional): Number of cross-validation folds.

    Methods:
        set_data: Setting the features and labels as model input.
    """
    def __init__(self, estimator, param_grid, scoring, cv):
        # Initialize the problem with the given parameters
        self.estimator = estimator
        self.params = param_grid
        self.scoring = scoring
        self.cv = cv

        # Calculate the binary representation length for each parameter in param_grid
        self.bit_len = list(map(lambda x: len(x).bit_length(), self.params.values()))
        self.seen_combinations = {}
        self.n_obj = len(self.scoring)

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
            for i, k in enumerate(Fs.keys()):
                f = -fvalues[i]
                Fs[k].append(f)
            Gs.append(gvalue)

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
            fvalues = [0 for _ in range(len(self.scoring))]
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
        cv_result = cross_validate(cls, self._x, self._y, cv=self.cv, scoring=self.scoring, n_jobs=self.cv)
        fvalues = []
        for v in list(cv_result.values())[2:]:
            fvalues += [np.mean(v)]
        return fvalues, 0

    # Initialize the result structures for objectives and constraints
    def _init_results(self):
        result = {}
        for i in range(len(self.scoring)):
            result[f'fvalue{int(i+1)}'] = []
        return result, []

    # Set x & y for training
    def set_data(self, x, y):
        self._x = x
        self._y = y