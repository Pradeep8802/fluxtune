import numpy as np

class NoExpertsLeftError(Exception):
    """
    Exception raised when no experts are left in the manager.
    """
    pass

class Expert:
    def __init__(self, model, name, penalty=1, predict_func=None, tol=0, eval_func=None):
        """
        Initialize the expert with a given model and name, with optional custom prediction and update functions.

        Parameters:
        - model: The predictive model used by the expert.
        - name: The name of the expert.
        - penalty: The penalty for an incorrect prediction.
        - predict_func: Optional custom prediction function.
        - tol: Tolerance for prediction error.
        - eval_func: Optional custom evaluation function.
        """
        self.model = model
        self.name = name
        self.penalty = penalty
        self.tol = tol
        self.mistakes = 0

        # Set the prediction function
        self.predict_func = predict_func if predict_func is not None else self._standard_predict

        # Set the evaluation function
        self.eval_func = eval_func if eval_func is not None else self._standard_eval

    def _standard_eval(self, X, y):
        """
        Default evaluation method that checks if the prediction is within the tolerance.

        Parameters:
        - X: Input data.
        - y: True value.

        Returns:
        - Penalty if the prediction is incorrect.
        """
        if abs(y - self.predict_func(X)) <= self.tol:
            return 0
        return self.penalty

    def _standard_update(self, X, y, penalty):
        """
        Default update method. Can be customized if needed.
        """
        pass

    def update_mistakes(self, mistakes):
        """
        Update the number of mistakes made by the expert.

        Parameters:
        - mistakes: Number of mistakes to add.
        """
        self.mistakes += mistakes

    def _standard_predict(self, X):
        """
        Default prediction method using the model's predict method.

        Parameters:
        - X: Input data.

        Returns:
        - Predicted values.
        """
        return self.model.predict(X)

    def predict(self, X):
        """
        Make a prediction using the expert's model.

        Parameters:
        - X: Input data.

        Returns:
        - Predicted values.
        """
        return self.predict_func(X)

    def eval(self, X, y, mistake_update=False):
        """
        Evaluate the expert's prediction and update mistakes if required.

        Parameters:
        - X: Input data.
        - y: True value.
        - mistake_update: Flag to indicate if mistakes should be updated.

        Returns:
        - Penalty for the prediction.
        """
        penalty = self.eval_func(X, y)
        if penalty != 0 and mistake_update:
            self.update_mistakes(penalty)
        self._standard_update(X, y, penalty)
        return penalty

class ExpertManager:
    def __init__(self, experts=None):
        """
        Initialize the ExpertManager with an optional list of experts.

        Parameters:
        - experts: A list of Expert instances.
        """
        if experts is None or len(experts) == 0:
            raise NoExpertsLeftError("No expert is made")
        
        self.experts = experts
        self.num_experts = len(self.experts)
        self.best_expert = experts[0]

        for i, expert in enumerate(experts):
            if not isinstance(expert, Expert):
                raise TypeError(f"Element at index {i} is not an instance of Expert")

    def add_expert(self, expert):
        """
        Add an expert to the manager.

        Parameters:
        - expert: An instance of the Expert class.
        """
        self.experts.append(expert)
        self.num_experts += 1

    def predict(self, X):
        """
        Get predictions from all managed experts.

        Parameters:
        - X: Input data.

        Returns:
        - A list of predictions from each expert.
        """
        return [expert.predict(X) for expert in self.experts]

    def experts_mistakes(self, show=False):
        """
        Get the number of mistakes made by each expert.

        Parameters:
        - show: Flag to print the mistakes.

        Returns:
        - List of mistakes for each expert.
        """
        if show:
            print('=============== Experts Mistakes ==========')
            for expert in self.experts:
                print(f'{expert.name} has made {expert.mistakes} mistakes')
            print('===========================================')
        return [expert.mistakes for expert in self.experts]

    def find_best_expert(self):
        """
        Find the expert with the least mistakes.

        Returns:
        - The expert with the least mistakes.
        """
        if not self.experts:
            return None
        return min(self.experts, key=lambda exp: exp.mistakes)

    def order_experts(self):
        """
        Order experts by the number of mistakes.

        Returns:
        - List of tuples with experts and their mistakes, ordered by the least mistakes.
        """
        ordered_experts = sorted(self.experts, key=lambda exp: exp.mistakes)
        return [(exp, exp.mistakes) for exp in ordered_experts]

    def remove_expert(self, index=0):
        """
        Remove an expert by index.

        Parameters:
        - index: Index of the expert to remove.
        """
        if index < len(self.experts):
            del self.experts[index]
            self.num_experts -= 1
        else:
            print("Expert not found in the manager")

    def eval_experts(self, X, y):
        """
        Evaluate all managed experts.

        Parameters:
        - X: Input data.
        - y: True value.
        """
        for expert in self.experts:
            expert.update_mistakes(expert.eval_func(X, y))
