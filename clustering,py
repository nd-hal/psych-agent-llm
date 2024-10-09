import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming 'train_data' is your DataFrame containing the categorical data

# One-hot encode all categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical = encoder.fit_transform(train_data)
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(train_data.columns))

# Standardize the encoded features
scaler = StandardScaler()
scaled_encoded_categorical_df = scaler.fit_transform(encoded_categorical_df)

# Apply UMAP for dimensionality reduction
umap = UMAP(n_components=3, n_neighbors=30, min_dist=0.3, random_state=42)
reduced_data = umap.fit_transform(scaled_encoded_categorical_df)

def test_k_values(reduced_data, max_k):
    silhouette_scores = []
    k_values = range(3, max_k + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_data)
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'For n_clusters = {k}, the average silhouette score is: {silhouette_avg:.4f}')
    return k_values, silhouette_scores

# Test k values from 2 to 15
k_values, silhouette_scores = test_k_values(reduced_data, 15)

import matplotlib.pyplot as plt

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.grid(True)
plt.show()
