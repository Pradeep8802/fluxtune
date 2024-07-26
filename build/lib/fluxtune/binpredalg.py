import numpy as np
from .experts import NoExpertsLeftError,Expert,ExpertManager

class Algorithm:
    """
    Base class for all algorithms.
    """
    def __init__(self, expertManager=None, max_mistakes=0, tol=0, verbose=True):
        self.verbose = verbose
        self.max_mistakes = max_mistakes
        self.tol = tol
        
        if expertManager is None:
            if self.verbose:
                print("No expert selected")
        elif isinstance(expertManager, ExpertManager):
            self.expertManager = expertManager
            if len(expertManager.experts) >= 1:
                self.best_expert = expertManager.experts[0]
            else:
                raise TypeError("Expected the ExpertManager to contain at least one expert")
        else:
            raise TypeError("Expected an instance of ExpertManager")

    def expertsMistakes(self):
        return self.expertManager.expertsMistakes(self.verbose)
    
    def log(self, message):
        if self.verbose:
            print(message)


class Naive(Algorithm):
    """
    Naive algorithm: Uses a single best expert for predictions.
    """
    def __init__(self, expertManager=None, max_mistakes=0, tol=0, verbose=True):
        super().__init__(expertManager=expertManager, max_mistakes=max_mistakes, tol=tol, verbose=verbose)
    
    def single_predict_alg(self, X):
        if self.best_expert is None:
            raise NoExpertsLeftError("No expert left") 

        self.best_expert = self.expertManager.best_expert
        return self.expertManager.best_expert.predict(X)

    def single_eval(self, X, y):
        if self.best_expert is None:
            raise NoExpertsLeftError("No expert left") 
        
        self.best_expert.eval(X, y, True)
        
        if self.best_expert.mistakes > self.max_mistakes:    
            self.expertManager.remove_expert(0)
            if len(self.expertManager.experts) >= 1:
                self.best_expert = self.expertManager.experts[0]
            else:
                raise NoExpertsLeftError("No expert left")
       
    def eval(self, X, y):
        self.log("Starting Naive evaluation method...")
        for time_instance in range(len(X)):
            self.single_eval(X[time_instance], y[time_instance])
        self.log("Naive evaluation method completed.")
    
    def predict_alg(self, X_array):
        results = []
        for X in X_array:
            try:
                result = self.single_predict_alg(X)
                results.append(result)
            except NoExpertsLeftError:
                raise NoExpertsLeftError('All experts have been exhausted, there is no best expert')
        return results


class MAJ(Algorithm):
    """
    Majority algorithm: Uses the majority vote of all experts for predictions.
    """
    def __init__(self, expertManager=None, max_mistakes=0, tol=0, verbose=True):
        super().__init__(expertManager=expertManager, max_mistakes=max_mistakes, tol=tol, verbose=verbose)
    
    def single_predict_alg(self, X):
        if not self.expertManager.experts:
            raise NoExpertsLeftError("No expert left")

        predictions = [expert.predict(X) for expert in self.expertManager.experts]
        majority_prediction = max(set(predictions), key=predictions.count)
        return majority_prediction
    
    def predict_alg(self, X_array):
        results = []
        for X in X_array:
            try:
                result = self.single_predict_alg(X)
                results.append(result)
            except NoExpertsLeftError:
                raise NoExpertsLeftError('All experts have been exhausted, there is no best expert')
        return results

    def single_eval(self, X, y):
        if not self.expertManager.experts:
            raise NoExpertsLeftError("No expert left")
    
        if isinstance(X, list):
            raise TypeError("Unhashable type: 'list'.\n Use eval method for more than one input.")

        prediction = self.single_predict_alg(X)

        if np.all(np.abs(np.asarray(prediction) - np.asarray(y)) <= self.tol):
            return

        exps_to_keep = []
        for expert in self.expertManager.experts:
            expert.eval(X, y, True)
            if expert.mistakes <= self.tol:
                exps_to_keep.append(expert)

        self.expertManager.experts = exps_to_keep
        self.best_expert = self.expertManager.find_best_expert()
        
        if self.best_expert is None:
            raise NoExpertsLeftError("No expert left")

    def eval(self, X, y):
        self.log("Starting MAJORITY Algorithm...")
        for time_instance in range(len(X)):
            self.single_eval(X[time_instance], y[time_instance])
        self.log("MAJORITY Algorithm completed.")


class WeightedMAJ(Algorithm):
    """
    Weighted Majority algorithm: Uses a weighted majority vote of all experts for predictions.
    """
    def __init__(self, expertManager=None, custom_weights=None, alpha=0.75, max_mistakes=0, tol=0, bound_experts=False, verbose=True):        
        super().__init__(expertManager=expertManager, max_mistakes=max_mistakes, tol=tol, verbose=verbose)
        self.alpha = alpha
        self.bound_experts = bound_experts

        if custom_weights is None:
            self.weights = np.array([1] * len(self.expertManager.experts))
        else:
            self.weights = np.array(custom_weights)
            if self.weights.shape[0] != len(self.expertManager.experts):
                raise TypeError(f"Expected weights to be of shape {len(self.expertManager.experts)} but got {self.weights.shape[0]}")
    
    def single_predict_alg(self, X):
        if not self.expertManager.experts:
            raise NoExpertsLeftError("No expert left")

        weighted_predictions = np.zeros(len(self.expertManager.experts))
        for idx, expert in enumerate(self.expertManager.experts):
            weighted_predictions[idx] = expert.predict(X) * self.weights[idx]

        return np.sum(weighted_predictions)
    
    def predict_alg(self, X_array):
        results = []
        for X in X_array:
            try:
                result = self.single_predict_alg(X)
                results.append(result)
            except NoExpertsLeftError:
                raise NoExpertsLeftError('All experts have been exhausted, there is no best expert')
        return results

    def single_eval(self, X, y):
        if not self.expertManager.experts:
            raise NoExpertsLeftError("No expert left")

        if isinstance(X, list):
           raise TypeError("Unhashable type: 'list'.\n Use multi_eval method for more than one input.")

        prediction = self.single_predict_alg(X)
    
        if np.all(np.abs(np.asarray(prediction) - np.asarray(y)) <= self.tol):
            return

        exps = []
        new_weights = []
        for idx, expert in enumerate(self.expertManager.experts):
            expert.eval(X, y, True)
            if self.bound_experts:
                if expert.mistakes <= self.tol:
                    exps.append(expert)
                    new_weights.append(self.weights[idx])
                else:
                    new_weights.append(self.weights[idx] * self.alpha)

            else:
                if expert.mistakes > self.tol:
                    self.weights[idx] *= self.alpha

        if(self.bound_experts):
                self.expertManager.experts = exps
                self.weights = np.array(new_weights)


        self.best_expert = self.expertManager.find_best_expert()
        if self.best_expert is None:
            raise NoExpertsLeftError("No expert left")

    def eval(self, X, y):
        self.log("Starting Weighted MAJORITY Algorithm...")
        for time_instance in range(len(X)):
            self.single_eval(X[time_instance], y[time_instance])
        self.log("Weighted MAJORITY Algorithm completed.")
