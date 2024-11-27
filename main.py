import numpy as np
import pulp
from numpy.linalg import svd
from scipy.io import loadmat, savemat
from helper import plot, save_plot

def optimize_matrix(X, W, M, latent_dim):
    m, n = X.shape
    prob = pulp.LpProblem("Optimize_matrix", pulp.LpMinimize)

    N = [[pulp.LpVariable(f"N_{k}_{j}", lowBound=0, upBound=5) for j in range(n)] for k in range(latent_dim)]
    t = [[pulp.LpVariable(f"t_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]
    s1 = [[pulp.LpVariable(f"s1_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]
    s2 = [[pulp.LpVariable(f"s2_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]

    prob += pulp.lpSum(W[i, j] * t[i][j] for i in range(m) for j in range(n))

    for i in range(m):
        for j in range(n):
            predicted = pulp.lpSum(M[i][k] * N[k][j] for k in range(latent_dim))
            prob += predicted <= 5

            if W[i, j] == 1:
                prob += t[i][j] - s1[i][j] == predicted - X[i, j]
                prob += t[i][j] - s2[i][j] == -(predicted - X[i, j])



    prob.solve()

    N_optimized = np.array([[N[k][j].varValue if N[k][j].varValue is not None else 0 for j in range(n)] for k in range(latent_dim)])
    return N_optimized

data = loadmat("data/movieratings.mat")

X = []
sums_errors = []

if 'X' in data:
    X = np.array(data['X'], dtype=np.float32)
    print("Loaded X matrix:")
    print(X)


W = np.array([[1 if x != 0 else 0 for x in row] for row in X])

m, n = X.shape

A, B, Vt = svd(X, full_matrices=False)

latent_dim = 2

U = []

if 'U0' in data:
    U = np.array(data['U0'], dtype=np.float32)
    print("Loaded U matrix:")
    print(U)
else:
    print("No U matrix")
    U = np.random.rand(m, latent_dim)

V = np.random.rand(latent_dim, n)

optimal = {
    "error": np.sum(np.abs(X - U@V)),
    "U": U,
    "V": V,
}

num_iterations = 200

for iteration in range(num_iterations):
    V = optimize_matrix(X, W, U, latent_dim)
    U = optimize_matrix(X.T, W.T, V.T, latent_dim).T

    errors = [[abs(X[i, j] - (U@V)[i][j]) if X[i, j] != 0 else 0 for j in range(n)] for i in range(m)]
    sum_errors = np.sum(errors)
    sums_errors.append(sum_errors)

    plot(sums_errors)

save_plot("plot.png")


print("X matrix is:")
print(X)

print("\nW matrix is:")
print(W)

print("\nOptimized U (user feature matrix):")
print(U)

print("\nOptimized V (movie feature matrix):")
print(V)

predicted_ratings = U@V
print("\nPredicted Ratings:")
print(predicted_ratings)

# savemat("result.mat", {"X": X, "U": U, "V": V, "diff": U@V - X})

print(f"\nOptimal latent dimension: {latent_dim}") # Number of latent features

# data = loadmat("result.mat")

# matrix = data["diff"]

# flat_indices = np.argsort(matrix.flatten())[-10:]

# coordinates = [np.unravel_index(idx, matrix.shape) for idx in flat_indices]

# top_values = [matrix[coord] for coord in coordinates]

# print("Coordinates of top 5 values:", coordinates)
# print("Top 5 values:", top_values)
