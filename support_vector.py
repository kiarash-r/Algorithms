import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# region fake data
d1= np.random.randn(20, 2)
d2= np.random.randn(20, 2)+ (3, 5)
data= np.vstack((d1, d2))
labels= np.array([0]* 20 + [1]* 20)
# endregion

model= SVC(kernel="linear")
model.fit(data, labels)

saport_vectors= model.support_vectors_  # Finding the boundary points

# Finding the main boundary line
w= model.coef_[0]
# Since y returns a 2-dimensional array, we pass 0 so that it returns only w0 and w1 themselves.
b= model.intercept_[0]

x_mrz= np.linspace(data[:, 0].min()-3, data[:, 0].max()+3, 4)  # Creating several x values to draw the boundary line
# ±3 in order to create a margin
f= lambda x_mrz: -(w[0]*x_mrz + b)/ w[1]  # Extracting y to draw the boundary line using the weights that the model has given us

y_mrz= f(x_mrz)

d_of_margin= 1/np.sqrt(np.sum(w**2)) # The distance from the separating line to the support vectors
y_margin_don= y_mrz- d_of_margin
y_margin_up= y_mrz+ d_of_margin

plt.scatter(data[:20, 0], data[:20, 1], color='green', marker='o')
plt.scatter(data[20:, 0], data[20:, 1], color='blue', marker='o')
plt.scatter(saport_vectors[:, 0], saport_vectors[:, 1], facecolors='none', edgecolors='red', s=75) # رسم هایلایت برای نقاط مرزی 
plt.plot(x_mrz, y_mrz, '-k') # Boundary line
plt.plot(x_mrz, y_margin_don, '--m')
plt.plot(x_mrz, y_margin_up, '--m')
plt.show()
