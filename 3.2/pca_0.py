import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import kmeans as km
from tqdm import tqdm

data, label = km.readdata()
label = np.array(label).astype(int)

pca_model = PCA(n_components=150)
pca_model.fit(data)
new_data = pca_model.transform(data)

subdata = new_data[0:3000]
sublabel = label[0:3000]
result_list = []
for k in tqdm(range(2,11)):
    k_model = KMeans(n_clusters=k,init='k-means++',random_state=0).fit(subdata)
    s = metrics.calinski_harabaz_score(subdata,k_model.labels_)
    result_list.append(s)
best_k = result_list.index(max(result_list)) + 2

t_1 = time.clock()
kmeans_model = KMeans(n_clusters=best_k, init='k-means++', random_state=0).fit(new_data)
predict = kmeans_model.labels_
t_2 = time.clock()
dt = t_2 - t_1
subsets = km.getsubset(kmeans_model.labels_,label,best_k)
prt = km.purity(subsets,best_k,len(label))
s = metrics.calinski_harabaz_score(new_data,kmeans_model.labels_)
print("Purity of k = %d is %lf, Calinski-Harabaz score is %lf, %lfsec used"%(best_k,prt,s,dt))


color_set = [[0.2157,0.4941,0.7216],[0.3020,0.6863,0.2902],
             [0.5961,0.3059,0.6392],[1,0.4980,0],[1,1,0.2],
             [0.6510,0.3373,0.1569],[0.9686,0.5059,0.7490],
             [0.6,0.6,0.6],[0.5,0.6,0.3],[0.6,0.2,0.4]]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(new_data)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(10,10))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i,0], X_norm[i,1], str(label[i]),color=color_set[predict[i]%10],fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()