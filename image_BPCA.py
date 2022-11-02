from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from im2col import gen_sliding_windows, make_blocks, revert_block
import matplotlib.pyplot as plt

X = np.array(Image.open("example.jpeg").convert("L"))

print("X => information")
print(f"shape of X: {X.shape}")
print(X)
print()

plt.imshow(X, cmap="Greys")
plt.savefig("X.jpeg")

gen_sliding_windows(X.shape[0], X.shape[1], 3, 3, 1)
blocks, coords = make_blocks(X)

X_blocks = np.array([x.reshape((9)) for x in blocks])

print("X_blocks information")
print(f"shape of X_blocks: {X_blocks.shape}")
print(X_blocks)
print()

scaler = MinMaxScaler()
X_rescaled = scaler.fit_transform(X_blocks)

print("X_rescaled information")
print(f"shape of X_rescaled: {X_rescaled.shape}")
print(X_rescaled)
print()

# 95% of variance
pca = PCA(n_components=0.95)
pca.fit(X_rescaled)
reduced = pca.transform(X_rescaled)

# print("informations:")
# print(f"shape of X: {reduced.shape}")
# print(f"explained variance ratio: {pca.explained_variance_ratio_}")
# print(f"explained variance : {pca.explained_variance_}")
# print(f"number components: {pca.n_components_}")
# print()

# reverting PCA
mu = np.mean(X_rescaled, axis=0)
n_comp = pca.n_components_
Xhat = np.dot(pca.transform(X_rescaled)[:, :n_comp], pca.components_[:n_comp, :])
Xhat += mu

Xhat = scaler.inverse_transform(Xhat)
Xhat = Xhat.astype("int")

print("Xhat information")
print(Xhat)
print(f"shape of Xhat: {Xhat.shape}")
print()

Xhat = np.array([x.reshape((3, 3)) for x in Xhat])

X_block_revert = revert_block(Xhat, coords, X.shape)
print(X_block_revert)
print(f"shape of X_block_revert: {X_block_revert.shape}")
print()

plt.imshow(X_block_revert, cmap="Greys")
plt.savefig("Xhat.jpeg")
