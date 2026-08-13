# Unsupervised Learning Models

## Table of Contents
1. [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)
2. [Clustering: Finding Groups in Data](#clustering)
3. [K-Means Clustering](#k-means-clustering)
4. [Elbow Method: Choosing the Right Number of Clusters](#elbow-method)
5. [Silhouette Analysis: Evaluating Cluster Quality](#silhouette-analysis)
6. [Other Clustering Algorithms](#other-clustering-algorithms)
7. [Distance Metrics and Similarity](#distance-metrics)
8. [Hands-On Examples](#hands-on-examples)

---

## Introduction to Unsupervised Learning

Unsupervised learning is a machine learning approach where we work with **unlabeled data**—data without predefined correct answers. The goal is to discover hidden patterns, structures, or relationships within the data.

### Supervised vs. Unsupervised Learning (Reminder)

| Aspect | Supervised | Unsupervised |
|--------|-----------|------------|
| Training Data | Labeled (has targets) | Unlabeled (no targets) |
| Goal | Predict known target | Discover patterns |
| Examples | Classification, Regression | Clustering, Dimensionality reduction |
| Use Cases | Spam detection, Price prediction | Customer segmentation, Data exploration |

### Why Unsupervised Learning?

1. **Labeling is expensive**: Getting labeled data requires time and experts
2. **Exploration**: Discover patterns you didn't know existed
3. **Understanding**: Learn the natural structure of your data
4. **Preprocessing**: Reduce data before supervised learning
5. **Outlier detection**: Find unusual cases

### Types of Unsupervised Learning

#### 1. **Clustering**
Grouping similar items together.

**Goal:** Partitions data into groups where items within groups are similar to each other, and different from other groups.

**Examples:**
- Customer segmentation (high-value, medium-value, low-value)
- Gene sequencing (grouping similar DNA sequences)
- Image compression (grouping similar colors)
- Document clustering (grouping similar topics)

#### 2. **Dimensionality Reduction**
Reducing the number of features while preserving important information.

**Goal:** Simplify data while keeping most of the useful information.

**Examples:**
- Principal Component Analysis (PCA)
- Feature reduction for visualization
- Noise reduction

#### 3. **Anomaly Detection**
Finding unusual or outlier cases.

**Examples:**
- Fraud detection in transactions
- Network intrusion detection
- Quality control in manufacturing
- Sensor fault detection

#### 4. **Association Rule Learning**
Finding relationships between variables.

**Example:** "Customers who buy bread often also buy milk" (Market basket analysis)

### Unsupervised Learning Workflow

<div align="center">
  <small><strong style="font-size: 12px;">Figura 1: Unsupervised Learning Workflow</strong></small><br>
   <img src="../assets/unsupervised_workflow.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

---

## Clustering: Finding Groups in Data

### What is Clustering?

Clustering is the process of grouping data points into clusters (groups) where:
- **Within-cluster similarity**: Points in the same cluster are very similar
- **Between-cluster dissimilarity**: Points in different clusters are very different

### Visual Example

<div align="center">
  <small><strong style="font-size: 12px;">Figura 2: Before and After Clustering</strong></small><br>
   <img src="../assets/clustering_before_after.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### How Does the Computer "Know" Similarity?

Computers measure distance between points. Points close together are similar:

**2D Example:**
```
Point A: (1, 2)
Point B: (1.2, 2.1) ← Very close to A
Point C: (10, 20)  ← Far from A

Distance A to B: ~0.22 (similar)
Distance A to C: ~19.4 (different)
```

### Clustering Algorithms Comparison

| Algorithm | Idea | Pros | Cons |
|-----------|------|------|------|
| K-Means | Partition into K clusters | Fast, simple | Need to specify K |
| Hierarchical | Build tree of clusters | Don't need K | Slower, memory intensive |
| DBSCAN | Density-based groups | Finds arbitrary shapes | Sensitive to parameters |
| Gaussian Mixture | Probabilistic clusters | Soft assignments | More complex |

### When to Use Clustering

✓ **Use when:**
- You want to discover natural groups
- Understand customer segments
- Organize data for analysis
- Reduce complexity
- Find outliers

✗ **Don't use when:**
- You already know the groups (use classification)
- Groups don't naturally exist in data
- You need precise, predefined categories

---

## K-Means Clustering

### What is K-Means?

K-Means is one of the most popular clustering algorithms. It partitions data into **K** clusters by minimizing the distance from each point to its cluster center.

### The K-Means Algorithm

**Intuitive Explanation:**
Imagine you want to place K warehouses to serve customers. You want to minimize total shipping distance.

**Algorithm Steps:**

<div align="center">
  <small><strong style="font-size: 12px;">Figura 3: K-Means Algorithm Steps</strong></small><br>
   <img src="../assets/kmeans_algorithm_steps.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Mathematical Definition

```
Goal: Minimize Within-Cluster Sum of Squares (WCSS)

WCSS = Σ(distance from point to its cluster center)²

For each cluster k:
  Σ |xᵢ - cₖ|²
  (sum squared distances from all points x to cluster center c)
```

### Key Concept: Inertia

**Inertia** is the sum of squared distances from each point to its nearest cluster center. Lower inertia = tighter clusters.

### Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
])

# Scale data (important!)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Create K-Means with K=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)

# Get results
labels = kmeans.labels_  # [0, 0, 1, 1, 0, 1]
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Cluster labels: {labels}")
print(f"Inertia: {inertia}")

# Predict for new point
new_point = np.array([[2, 2]])
new_point_scaled = scaler.transform(new_point)
prediction = kmeans.predict(new_point_scaled)
print(f"New point assigned to cluster: {prediction}")
```

### Why Scale Data?

K-Means uses distance calculations. Without scaling, features with larger ranges dominate:

```
Example: Salary (0-100,000) vs Age (0-100)
Without scaling: Salary differences drown out age differences

Before scaling:
Distance = √[(50000)² + (25)²] ≈ 50,000

After scaling (0-1):
Distance = √[(0.5)² + (0.25)²] ≈ 0.56
Both features equally important!
```

### Advantages and Disadvantages

**Advantages:**
- ✓ Simple and fast
- ✓ Scales well to large datasets
- ✓ Easy to understand
- ✓ Works well with spherical clusters

**Disadvantages:**
- ✗ Must specify K in advance
- ✗ Sensitive to initialization (random centers)
- ✗ Works poorly with non-spherical clusters
- ✗ Sensitive to outliers
- ✗ Each point belongs to exactly one cluster

### Common Pitfalls

**Different Random Seeds Give Different Results:**
```python
# Run 1: Different result from Run 2
kmeans1 = KMeans(n_clusters=3, random_state=42)
kmeans2 = KMeans(n_clusters=3, random_state=43)

# Solution: Use random_state for reproducibility
# Or run multiple times and pick best result
```

**Wrong Number of Clusters:**
```
K=2: Merges natural groups
K=4: Splits natural groups
K=3: Just right!
```

---

## Elbow Method: Choosing the Right Number of Clusters

### The Problem

**How do we know how many clusters (K) to choose?**

If K is too small, we lose information. If K is too large, we overfit.

```
K=1: All points in one cluster (no segmentation)
K=10: Each point almost its own cluster (overfitting)
K=3: Just right! (natural grouping)
```

### The Elbow Method Idea

As we increase K:
- Inertia always decreases (adding more clusters fits tighter)
- But the improvement diminishes

<div align="center">
  <small><strong style="font-size: 12px;">Figura 4: Elbow Method</strong></small><br>
   <img src="../assets/elbow_method.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Finding the Elbow

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Try different values of K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Plot
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid()
plt.show()

# Look for the elbow point (where slope changes)
```

### Interpreting the Elbow Plot

<div align="center">
  <small><strong style="font-size: 12px;">Figura 5: Interpreting Elbow Plots</strong></small><br>
   <img src="../assets/elbow_scenarios.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Elbow Method Limitations

❌ **Doesn't always give clear answer**
❌ **Subjective (where's the elbow?)**
❌ **Only considers cluster tightness, not separation**

**Better approach:** Combine with Silhouette Analysis!

---

## Silhouette Analysis: Evaluating Cluster Quality

### What is Silhouette Score?

The Silhouette Score measures how well each point fits in its cluster. It ranges from -1 to +1:

- **+1**: Point is well-clustered
- **0**: Point is near cluster boundary
- **-1**: Point is in wrong cluster

### How It Works

For each point, calculate:
```
a = Average distance to other points in same cluster
b = Average distance to nearest cluster (other than own)

Silhouette Score = (b - a) / max(a, b)
```

**Interpretation:**
- If b >> a: Point is in right cluster (positive score)
- If a ≈ b: Point is near boundary (score near 0)
- If a > b: Point might be in wrong cluster (negative score)

### Visual Example

<div align="center">
  <small><strong style="font-size: 12px;">Figura 6: Silhouette Score Intuition</strong></small><br>
   <img src="../assets/silhouette_concept.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Computing Silhouette Score

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# Assuming kmeans model is already fitted
silhouette_avg = silhouette_score(data, kmeans.labels_)
print(f"Average Silhouette Score: {silhouette_avg:.3f}")

# Get score for each point
silhouette_vals = silhouette_samples(data, kmeans.labels_)

# Visualize
fig, ax = plt.subplots()
y_lower = 10

for i in range(kmeans.n_clusters):
    cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == i]
    cluster_silhouette_vals.sort()
    
    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster
    
    ax.fill_between(range(y_lower, y_upper), cluster_silhouette_vals)
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster Label')
plt.show()
```

### Interpreting Silhouette Plots

<div align="center">
  <small><strong style="font-size: 12px;">Figura 7: Silhouette Plot Examples</strong></small><br>
   <img src="../assets/silhouette_plots.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Silhouette Score Ranges

| Score | Interpretation |
|-------|-----------------|
| 0.71-1.00 | Strong structure |
| 0.51-0.70 | Reasonable structure |
| 0.26-0.50 | Weak structure |
| ≤ 0.25 | No substantial structure |

### Using Silhouette to Choose K

```python
from sklearn.metrics import silhouette_score

# Try different K values
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.show()

# Choose K with highest silhouette score
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K: {best_k}")
```

### Advantages Over Elbow Method

✓ **More objective** (numerical score)  
✓ **Considers both cohesion and separation**  
✓ **Per-point scores** (see which points are problematic)

---

## Other Clustering Algorithms

### 1. Hierarchical Clustering

**Idea:** Build a tree (dendrogram) of clusters by successively merging or splitting clusters.

**Two Approaches:**

**Agglomerative (Bottom-Up):**

**Divisive (Top-Down):**

<div align="center">
  <small><strong style="font-size: 12px;">Figura 8: Hierarchical Clustering Approaches</strong></small><br>
   <img src="../assets/hierarchical_clustering.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

**Implementation:**
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Compute linkage matrix
Z = linkage(data, method='ward')  # 'ward', 'complete', 'average', 'single'

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cut tree at certain level to get clusters
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=2, criterion='maxclust')  # t=2 means 2 clusters
```

**Advantages:**
- Don't need to specify K upfront (can cut at any level)
- Dendrogram shows relationships
- Doesn't assume spherical clusters

**Disadvantages:**
- Slower O(n²)
- Greedy (can't undo merges)
- Results depend on linkage method

### 2. DBSCAN (Density-Based Spatial Clustering)

**Idea:** Clusters are dense regions separated by sparse regions.

<div align="center">
  <small><strong style="font-size: 12px;">Figura 9: DBSCAN Density and Noise</strong></small><br>
   <img src="../assets/dbscan_density.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

**Key Concepts:**
- **ε (eps)**: Maximum distance between points
- **minPts**: Minimum points in neighborhood

**Implementation:**
```python
from sklearn.cluster import DBSCAN

# ε and minPts are important parameters
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(data)

# Label -1 means noise/outlier
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

**Advantages:**
- Finds arbitrary cluster shapes
- Discovers outliers
- Don't need to specify number of clusters

**Disadvantages:**
- Hard to choose ε and minPts
- Struggles with varying density
- Slower for large datasets

### 3. Gaussian Mixture Models (GMM)

**Idea:** Data comes from mixture of Gaussian distributions.

```
Instead of hard assignment:
  Point belongs to Cluster 1 or Cluster 2

Use soft assignment:
  Point has 70% chance of Cluster 1
  Point has 30% chance of Cluster 2
```

**Implementation:**
```python
from sklearn.mixture import GaussianMixture

# Specify number of components
gmm = GaussianMixture(n_components=3)
gmm.fit(data)

# Hard assignments
labels = gmm.predict(data)

# Soft assignments (probabilities)
probabilities = gmm.predict_proba(data)
# [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], ...]

# BIC/AIC for choosing n_components
bic_scores = [GaussianMixture(n).fit(data).bic(data) for n in range(1, 10)]
```

**Advantages:**
- Probabilistic (soft assignments)
- Works with non-spherical clusters
- BIC/AIC for model selection

**Disadvantages:**
- More parameters to estimate
- Slower
- Assumes Gaussian distributions

---

## Distance Metrics and Similarity

### Why Distance Matters

Different distance metrics can give different clustering results!

<div align="center">
  <small><strong style="font-size: 12px;">Figura 10: Euclidean vs Manhattan Distance</strong></small><br>
   <img src="../assets/distance_metrics.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Common Distance Metrics

#### 1. **Euclidean Distance**
Straight-line distance (most common).

```
d = √[(x₂-x₁)² + (y₂-y₁)² + ...]

Example: Points (1,2) and (4,6)
d = √[(4-1)² + (6-2)²] = √[9 + 16] = 5
```

**When to use:** Most general purpose clustering

#### 2. **Manhattan Distance (L1)**
Distance along grid (city blocks).

```
d = |x₂-x₁| + |y₂-y₁| + ...

Example: Points (1,2) and (4,6)
d = |4-1| + |6-2| = 3 + 4 = 7
```

**When to use:** Categorical data, sparse data

#### 3. **Cosine Similarity**
Measures angle between vectors (not magnitude).

```
similarity = (A·B) / (|A||B|)

Ranges from -1 (opposite) to 1 (same direction)
```

**When to use:** Text data, high-dimensional data

#### 4. **Jaccard Similarity**
Overlap between sets.

```
J = |A ∩ B| / |A ∪ B|

Example:
A = {1, 2, 3}
B = {2, 3, 4}
J = 2/4 = 0.5
```

**When to use:** Categorical/binary data, sets

### Implementation

```python
from sklearn.metrics.pairwise import (
    euclidean_distances, 
    manhattan_distances, 
    cosine_distances
)

# Example data
X = [[0, 0], [1, 1], [10, 10]]

# Calculate distances
euc = euclidean_distances(X)
manhattan = manhattan_distances(X)
cosine = cosine_distances(X)

print("Euclidean:\n", euc)
```

---

## Hands-On Examples

### Example 1: Customer Segmentation with K-Means

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Sample customer data
data = pd.DataFrame({
    'Annual_Income': [30000, 50000, 25000, 80000, 120000, 70000],
    'Spending_Score': [10, 50, 15, 90, 95, 85]
})

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow method + Silhouette
silhouette_scores = []
inertias = []
K_range = range(2, 6)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Choose best K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K: {best_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
data['Segment'] = kmeans_final.fit_predict(data_scaled)

print("\nCustomer Segments:")
for segment in range(best_k):
    print(f"\nSegment {segment}:")
    print(data[data['Segment'] == segment])

# Visualize
plt.scatter(data['Annual_Income'], data['Spending_Score'], 
           c=data['Segment'], cmap='viridis', s=100)
plt.scatter(scaler.inverse_transform(kmeans_final.cluster_centers_)[:, 0],
           scaler.inverse_transform(kmeans_final.cluster_centers_)[:, 1],
           c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

### Example 2: Comparing DBSCAN and K-Means

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate non-spherical data
X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# K-Means
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')
ax1.set_title(f'K-Means (K=4)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# DBSCAN
unique_labels = set(y_dbscan)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Noise in black
    class_member_mask = (y_dbscan == k)
    xy = X_scaled[class_member_mask]
    ax2.scatter(xy[:, 0], xy[:, 1], c=[col], s=50)
ax2.set_title('DBSCAN')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### Example 3: Using Silhouette Analysis to Find K

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Generate data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, n_clusters in enumerate([2, 3, 4, 5]):
    ax = axes[idx // 2, idx % 2]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    silhouette_avg = silhouette_score(X, labels)
    silhouette_vals = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()
        
        size_cluster = cluster_vals.shape[0]
        y_upper = y_lower + size_cluster
        
        ax.fill_betweenx(range(y_lower, y_upper), cluster_vals)
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title(f"K={n_clusters}, Score={silhouette_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")

plt.tight_layout()
plt.show()
```

---

## Summary

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| K-Means | Spherical clusters | Fast, simple | Need K, sensitive to init |
| Hierarchical | Tree structure | No need K | Slow, memory intensive |
| DBSCAN | Arbitrary shapes | Finds outliers | Hard params, varying density |
| GMM | Soft assignments | Probabilistic | More complex |

### Choosing Algorithm Flowchart

<div align="center">
  <small><strong style="font-size: 12px;">Figura 11: Choosing a Clustering Algorithm</strong></small><br>
   <img src="../assets/clustering_algorithm_flowchart.svg"/><br>
  <small style="margin-top: 4px; font-size: 10px;">Fonte: Material produzido pelos autores (2026).</small><br>
</div>

### Key Takeaways

1. **Always scale your data** before clustering
2. **Use multiple methods** to validate results (Elbow + Silhouette)
3. **Visualize clusters** whenever possible
4. **Consider domain knowledge** - statistics alone doesn't determine right K
5. **Start with K-Means**, then try others if needed

---

## Recommended Videos

- **Clustering Basics**: https://www.youtube.com/watch?v=gs9E7E0qOIc
- **Elbow Method & Silhouette Analysis**: https://www.youtube.com/watch?v=R2e3Ls9H_fc

---

## Additional Resources

- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Understanding K-Means Clustering](https://www.datacamp.com/community/tutorials/k-means-clustering-python)
- [Silhouette Analysis Guide](https://en.wikipedia.org/wiki/Silhouette_(clustering))

---

**Congratulations!** You've now mastered both supervised and unsupervised learning. Ready to build real-world projects combining these techniques!
