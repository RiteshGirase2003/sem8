# 📦 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # For visualization
from sklearn.preprocessing import StandardScaler

# 📥 Step 1: Load the Iris dataset directly from sklearn
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
feature_names = iris.feature_names

# 🎯 Optional: View as a DataFrame (for clarity)
df = pd.DataFrame(X, columns=feature_names)
print("Sample Data:")
print(df.head())

# ✨ Step 2: Scale the features to bring all features to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔄 Step 3: Apply KMeans clustering (we know Iris has 3 classes)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 🧪 Step 4: Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# 📈 Step 5: Plot the clustered data
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering on Iris Dataset (Visualized using PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.show()
