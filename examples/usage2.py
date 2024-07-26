import numpy as np
from fluxtune.experts import Expert, ExpertManager
from fluxtune.binpredalg import MAJ
class model:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, X):
        return X * self.factor

# Create multiple experts
expert1 = Expert(model(1), name="Expert1")
expert2 = Expert(model(2), name="Expert2")
expert3 = Expert(model(3), name="Expert3")

# Create an ExpertManager
expert_manager = ExpertManager(experts=[expert1, expert2, expert3])

# Initialize the MAJ algorithm
maj_alg = MAJ(expertManager=expert_manager, max_mistakes=2, tol=1, verbose=True)

# Generate some data
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])  # True values

# Evaluate and predict
maj_alg.eval(X, y)
predictions = maj_alg.predict_alg(X)

print("Predictions:", predictions)
