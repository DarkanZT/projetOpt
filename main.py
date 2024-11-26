import numpy as np
import pulp
from numpy.linalg import svd
from scipy.io import loadmat

def optimize_matrix(X, W, M, latent_dim):
    m, n = X.shape
    prob = pulp.LpProblem("Optimize_matrix", pulp.LpMinimize)

    # Decision variables
    N = [[pulp.LpVariable(f"N_{k}_{j}", lowBound=0, upBound=5) for j in range(n)] for k in range(latent_dim)]
    t = [[pulp.LpVariable(f"t_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]
    s1 = [[pulp.LpVariable(f"s1_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]
    s2 = [[pulp.LpVariable(f"s2_{i}_{j}", lowBound=0) for j in range(n)] for i in range(m)]

    # Objective function: Minimize weighted absolute error
    prob += pulp.lpSum(W[i, j] * t[i][j] for i in range(m) for j in range(n))

    # Constraints
    for i in range(m):
        for j in range(n):
            if W[i, j] == 1:  # Only apply to rated entries
                predicted = pulp.lpSum(M[i][k] * N[k][j] for k in range(latent_dim))

                # Absolute error constraints
                prob += t[i][j] - s1[i][j] == predicted - X[i, j]
                prob += t[i][j] - s2[i][j] == -(predicted - X[i, j])

    # Solve the problem
    prob.solve()

    # Extract optimized V matrix
    N_optimized = np.array([[N[k][j].varValue for j in range(n)] for k in range(latent_dim)])
    return N_optimized
# Alternating optimization
data = loadmat("data/smallexample.mat")

X = []

if 'X' in data:
    X = data['X']
    print("Loaded X matrix:")
    print(X)


# Weight matrix W (1 if rated, 0 otherwise)
W = np.array([[1 if x != 0 else 0 for x in row] for row in X])

m, n = X.shape  # Number of users and movies

A, B, Vt = svd(X, full_matrices=False)

# Compute cumulative explained variance
explained_variance = np.cumsum(B) / np.sum(B)
print("Explained variance:", explained_variance)

# Choose k based on 90% variance explained
latent_dim = np.argmax(explained_variance >= 0.9 ) + 1
print(latent_dim)

# Initialize U and V randomly
U = []

if 'U' in data:
    U = data['U']
    print("Loaded U matrix:")
    print(U)
else:
    U = np.random.rand(m, latent_dim)

V = np.random.rand(latent_dim, n)

num_iterations = 10
for iteration in range(num_iterations):
    V = optimize_matrix(X, W, U, latent_dim)
    U = optimize_matrix(X.T, W.T, V.T, latent_dim).T

    print(f"Iteration {iteration + 1} complete.")

# Final results
print("X matrix is:")
print(X)

print("\nW matrix is:")
print(W)

print("\nOptimized U (user feature matrix):")
print(U)

print("\nOptimized V (movie feature matrix):")
print(V)

# Predicted ratings
predicted_ratings = U.__matmul__(V)
print("\nPredicted Ratings:")
print(predicted_ratings)


print(f"\nOptimal latent dimension: {latent_dim}") # Number of latent features