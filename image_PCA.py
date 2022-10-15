from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

X = np.array(Image.open("example.jpeg").convert("L"))
X_rescaled = X

print(f"shape of X: {X.shape}")
print()

scaler = MinMaxScaler()
X_rescaled = scaler.fit_transform(X)

# 95% of variance
pca = PCA(n_components=0.95)
pca.fit(X_rescaled)
reduced = pca.transform(X_rescaled)

print("informations:")
print(f"shape of X: {reduced.shape}")
print(f"explained variance ratio: {pca.explained_variance_ratio_}")
print(f"explained variance : {pca.explained_variance_}")
print(f"number components: {pca.n_components_}")
print()

# using number components
# pca_cmp = PCA(n_components=3)
# pca_cmp.fit(X_rescaled)
# reduced_cmp = pca_cmp.transform(X_rescaled)

# reverting PCA
mu = np.mean(X_rescaled, axis=0)
n_comp = pca.n_components_
Xhat = np.dot(pca.transform(X)[:, :n_comp], pca.components_[:n_comp, :])
Xhat += mu

print(f"Xhat: {Xhat.shape}")
