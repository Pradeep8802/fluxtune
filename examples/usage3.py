import numpy as np
from fluxtune.experts import Expert, ExpertManager
from fluxtune.binpredalg import WeightedMAJ
class model:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, X):
        return X * self.factor

# Defining experts
expert1 = Expert(model(1), name="Expert1")
expert2 = Expert(model(2), name="Expert2")
expert3 = Expert(model(3), name="Expert3")

# Creating an ExpertManager
expert_manager = ExpertManager(experts=[expert1, expert2, expert3])

# Initializing the WeightedMAJ algorithm
weights = [0.5, 0.3, 0.2]
weighted_maj_alg = WeightedMAJ(expertManager=expert_manager, custom_weights=weights, alpha=0.9, max_mistakes=3, tol=1, bound_experts=True, verbose=True)

# data
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])  

# Evaluation
weighted_maj_alg.eval(X, y)

# prediction
predictions = weighted_maj_alg.predict_alg(X)

print("Predictions:", predictions)
