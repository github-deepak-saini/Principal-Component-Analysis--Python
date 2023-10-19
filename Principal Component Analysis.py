
# Principal Component Analysis (PCA) for Dimensionality Reduction on Iris Dataset

# import matplotlib
import matplotlib.pyplot as plt
# import sklearn as sklearn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn import datasets


irisset = datasets.load_iris()
X = irisset.data
Y = irisset.target

pca = PCA(n_components=2);
Xp = pca.fit(X).transform(X) 
# Should this be pca.fit(X).transform(X) or be pca.transform(X).fit(X) ?

# %matplotlib inline
plt.figure(1)
plt.scatter(Xp[:,0], Xp[:,1], c=Y, cmap='jet', s=10)
plt.suptitle('Original Clusters')
plt.grid(1, which='both')
plt.axis('tight')
plt.show()

GMM = GaussianMixture(n_components=3)
#OMP_NUM_THREADS=1
GMM.fit(Xp)
y_predG = GMM.predict(Xp)

# %matplotlib inline
plt.figure(2)
plt.scatter(Xp[:,0], Xp[:,1], c=y_predG, cmap='jet', s=10)
plt.suptitle('GMM Clusters')
plt.grid(1, which='both')
plt.axis('tight')
plt.show()

cmat = confusion_matrix(Y,y_predG)
