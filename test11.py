import numpy as np

class UnifiedGradient:
    def __init__(self, decoupling_factor, balance_factor):
        self.decoupling_factor = decoupling_factor
        self.balance_factor = balance_factor

    def compute_normals(self, data):
        normals = np.gradient(data)
        return normals

    def moment_balance(self, forces, positions):
        moment = np.cross(positions, forces)
        return np.sum(moment)

    def decoupled_balance(self, state):
        return state * self.decoupling_factor

    def apply_gradient(self, field):
        normals = self.compute_normals(field)
        balanced_field = self.decoupled_balance(field) + normals
        return balanced_field

# Example Usage:
if __name__ == "__main__":
    ug = UnifiedGradient(decoupling_factor=0.5, balance_factor=1.0)
    sample_data = np.array([1, 2, 3, 4, 5])
    result = ug.apply_gradient(sample_data)
    print("Balanced Field:", result)