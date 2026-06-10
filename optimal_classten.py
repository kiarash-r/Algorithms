#Sometimes we dont even know how many classes we have in unsupervised classification
#This is one way to find the optimal number of classes

import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# region make data
number_of_data= 84
c1= np.random.randn(number_of_data, 2)+ (-8, -1)
c2= np.random.randn(number_of_data, 2)+ (0, 1)
c3= np.random.randn(number_of_data, 2)+ (5, 6)
data= np.vstack((c1, c2, c3))
plt.plot(data[:, 0], data[:, 1], "ok")
plt.show()
# endregion

class_score= []
num_of_class= range(2, 9)
for k in num_of_class:
    model= KMeans(n_clusters= k)
    labels= model.fit_predict(data)
    score= silhouette_score(data, labels)
    class_score.append(score)

plt.bar(num_of_class, class_score)
plt.show()

best_K= num_of_class[np.argmax(class_score)]

final_model= KMeans(n_clusters= best_K)
final_lbls= final_model.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=final_lbls, cmap="viridis")
plt.show()
