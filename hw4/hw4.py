import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    with open(filepath,mode='r') as file:
        reader = csv.DictReader(file)
        data=[dict(row) for row in reader]
    return data

def calc_features(row):
    features = np.array([
        float(row['Population']),
        float(row['Net migration']),
        float(row['GDP ($ per capita)']),
        float(row['Literacy (%)']),
        float(row['Phones (per 1000)']),
        float(row['Infant mortality (per 1000 births)'])
    ], dtype=np.float64)
    return features
    
def hac(features):
    n = len(features)
    # Create the initial distance matrix
    dist_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dist_matrix[j, i] = np.linalg.norm(features[i] - features[j])

    # Prepare to record the history of merges in the Z matrix
    Z = np.zeros((n - 1, 4))
    cluster_sizes = np.ones(n)

    # An array to track which cluster each point belongs to
    active_clusters = np.arange(n)

    for k in range(n - 1):
        # Find the pair of clusters with the smallest distance
        min_idx = np.argmin(dist_matrix)
        i, j = divmod(min_idx, n)

        # Record in Z: smaller index first, merge distance, new cluster size
        Z[k, 0] = min(active_clusters[i], active_clusters[j])
        Z[k, 1] = max(active_clusters[i], active_clusters[j])
        Z[k, 2] = dist_matrix[i, j]
        Z[k, 3] = cluster_sizes[i] + cluster_sizes[j]

        # Update cluster sizes and deactivate the merged cluster
        cluster_sizes[i] = Z[k, 3]
        cluster_sizes[j] = 0  # Deactivate the merged cluster

        # Update the active cluster indices
        new_cluster_index = max(active_clusters) + 1
        active_clusters[i] = new_cluster_index
        active_clusters[j] = new_cluster_index

        # Update the distance matrix
        for l in range(n):
            if l != i and l != j:
                dist_matrix[i, l] = dist_matrix[l, i] = min(dist_matrix[i, l], dist_matrix[j, l])
        dist_matrix[:, j] = dist_matrix[j, :] = np.inf

    return Z
# Previous version from HW4_Coding, successful but complex
# def hac(features):
#     n = len(features)
#     Z = np.zeros((n-1, 4))
    
#     # Initialize distance matrix
#     d = np.zeros((n, n))
#     for i in range(n):
#         for j in range(0,i):
#             d[i, j] = np.linalg.norm(features[i] - features[j])
#             d[j, i] = d[i, j]
#     cluster = {i: [i] for i in range(n)}

#     ad = np.ones((2 * n, 2 * n)) * math.inf
#     for i in range(n):
#         for j in range(i+1,n):
#             ad[i,j]=d[i,j]
#     for c in range(n - 1): # Merge step
#         index = np.argmin(ad)
#         (row, col) = np.unravel_index(index, ad.shape) # Update distance
#         for i in range(n + c):
#             ad[i, n + c] = min(ad[min(i, row), max(i, row)], ad[min(i, col),max(col, i)])
#         cluster[n + c] = cluster[row] + cluster[col] # Update Z
#         Z[c, 0] = row
#         Z[c, 1] = col
#         Z[c, 2] = ad[row, col] 
#         Z[c, 3] = len(cluster[n + c])
#         ad[row, :] = math.inf
#         ad[:, row] = math.inf
#         ad[col, :] = math.inf
#         ad[:, col] = math.inf
#     cluster
#     return Z

def fig_hac(Z, names):
    fig=plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    plt.show()
    return fig

def normalize_features(features):
    features = np.array(features)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    normalize_features=(features - min_vals) / (max_vals - min_vals)
    return [np.array(row, dtype=np.float64) for row in normalize_features]


if __name__ == "__main__":
    data = load_data('countries.csv')
    country_names = [row['Country'] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n=50
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    fig_hac(Z_raw, country_names[:n])
    plt.show()
    
