"""
Example shows usage of UnsupervisedRandomForest (URerF) class.

This demonstrates how to use URerF for unsupervised learning tasks such as
clustering and generating similarity matrices from unlabeled data.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import adjusted_rand_score

from rerf.urerf import UnsupervisedRandomForest

# Example 1: Basic usage with make_classification
print("=" * 60)
print("Example 1: Basic URerF with make_classification dataset")
print("=" * 60)

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Create an Unsupervised Random Forest
# You can use either "RerF" (Randomer Forest) or "Base" (Random Forest)
clf = UnsupervisedRandomForest(
    projection_matrix="RerF",  # Use "RerF" for better performance
    n_estimators=100,
    random_state=0,
)

# Fit the model (note: y is not used, but we show it for reference)
print("\nFitting URerF model...")
clf.fit(X)

# Transform the data into a similarity matrix
print("Generating similarity matrix...")
sim_mat = clf.transform()

print(f"Similarity matrix shape: {sim_mat.shape}")
print(f"Similarity matrix diagonal (should be all 1s): {sim_mat.diagonal()[:5]}...")

# Use the similarity matrix for clustering
print("\nPerforming hierarchical clustering...")
cluster = AgglomerativeClustering(n_clusters=2)
predict_labels = cluster.fit_predict(sim_mat)

# Evaluate clustering performance
score = adjusted_rand_score(y, predict_labels)
print(f"Adjusted Rand Score: {score:.4f}")
print("(Score closer to 1.0 indicates better clustering performance)")

# Example 2: Using Base Random Forest (standard RF)
print("\n" + "=" * 60)
print("Example 2: URerF with 'Base' projection matrix")
print("=" * 60)

clf_base = UnsupervisedRandomForest(
    projection_matrix="Base",  # Standard Random Forest
    n_estimators=100,
    random_state=0,
)

clf_base.fit(X)
sim_mat_base = clf_base.transform()

cluster_base = AgglomerativeClustering(n_clusters=2)
predict_labels_base = cluster_base.fit_predict(sim_mat_base)

score_base = adjusted_rand_score(y, predict_labels_base)
print(f"Adjusted Rand Score (Base): {score_base:.4f}")

# Example 3: Visualizing the similarity matrix
print("\n" + "=" * 60)
print("Example 3: Visualizing similarity matrices")
print("=" * 60)

# Create a smaller dataset for better visualization
X_small, y_small = make_blobs(
    n_samples=100, centers=3, n_features=2, random_state=42
)

clf_small = UnsupervisedRandomForest(
    projection_matrix="RerF", n_estimators=50, random_state=42
)
clf_small.fit(X_small)
sim_mat_small = clf_small.transform()

# Sort by true labels for better visualization
sorted_indices = np.argsort(y_small)
sorted_sim_mat = sim_mat_small[sorted_indices][:, sorted_indices]

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot similarity matrix
im1 = axes[0].imshow(sorted_sim_mat, cmap="viridis")
axes[0].set_title("URerF Similarity Matrix\n(sorted by true labels)")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Sample index")
plt.colorbar(im1, ax=axes[0])

# Plot data points colored by true labels
axes[1].scatter(X_small[:, 0], X_small[:, 1], c=y_small, cmap="viridis")
axes[1].set_title("Original Data (colored by true labels)")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.tight_layout()
# Save in the examples directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "urerf_example.png")
try:
    plt.savefig(output_path, dpi=100)
    print(f"Similarity matrix visualization saved to {output_path}")
except (IOError, OSError) as e:
    print(f"Warning: Could not save visualization: {e}")
finally:
    plt.close(fig)

# Example 4: Parameter tuning
print("\n" + "=" * 60)
print("Example 4: Exploring different parameters")
print("=" * 60)

# Test different numbers of estimators
n_estimators_list = [10, 50, 100, 200]
scores = []

for n_est in n_estimators_list:
    clf_temp = UnsupervisedRandomForest(
        projection_matrix="RerF",
        n_estimators=n_est,
        random_state=0,
    )
    clf_temp.fit(X)
    sim_mat_temp = clf_temp.transform()
    cluster_temp = AgglomerativeClustering(n_clusters=2)
    pred_labels_temp = cluster_temp.fit_predict(sim_mat_temp)
    score_temp = adjusted_rand_score(y, pred_labels_temp)
    scores.append(score_temp)
    print(f"n_estimators={n_est:3d} -> ARI: {score_temp:.4f}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("URerF (Unsupervised Randomer Forest) can be used to:")
print("1. Generate similarity matrices from unlabeled data")
print("2. Support downstream clustering tasks")
print("3. Capture complex relationships in high-dimensional data")
print("\nKey parameters:")
print("- projection_matrix: 'RerF' or 'Base'")
print("- n_estimators: Number of trees (more trees = better but slower)")
print("- max_features: Number of features to consider at each split")
print("- feature_combinations: For RerF, number of features to combine")
print("=" * 60)
