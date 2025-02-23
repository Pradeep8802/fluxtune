import numpy as np
from fluxtune.experts import Expert, ExpertManager
from fluxtune.binpredalg import Naive


class model:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, X):
        return X * self.factor

# Defining experts
# singlex expert
expert1 = Expert(model(1), name="Expert1")

# Creating an ExpertManager
expert_manager = ExpertManager(experts=[expert1])

# Initialize the Naive algorithm
naive_alg = Naive(expertManager=expert_manager, max_mistakes=1, tol=0, verbose=True)

# data
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5]) 

# Evaluation
naive_alg.eval(X, y)

# prediction
predictions = naive_alg.predict_alg(X)

print("Predictions:", predictions)

