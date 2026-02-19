import matplotlib.pyplot as plt
import numpy as np

#region maked fake data
num_of_data= 50
cluster1= np.random.randn(num_of_data, 2)+ (2, 4)
cluster2= np.random.randn(num_of_data, 2)+ (8, 9)
target_labels= np.vstack((np.zeros((num_of_data, 1)), np.ones((num_of_data, 1))))
data= np.vstack((cluster1, cluster2))
fake_x= data[:, 0]
fake_y= data[:, 1]

plt.scatter(fake_x, fake_y, marker='o', c='black')
plt.show()
#endregion

f1= lambda x: 1/ (1+ np.exp(-x)) # log_sigmoid
df1= lambda x: f1(x)* (1- f1(x))
f2= lambda x: 1/ (1+ np.exp(-x)) 
df2= lambda x: f1(x)* (1- f1(x))

w1= np.array([[0.2, 0.3], [0.25, 0.35]])
b1= np.array([0.5, 0.5]).reshape(2, 1)
w2= np.array([0.25, 0.35]).reshape(1, 2)
b2= np.array([0.4]).reshape(1, 1)
A= 0.1

for epoc in range(1000):
    T_error= 0
    for i in range(len(data)):
        a0= data[i].reshape(-1, 1)
        target= target_labels[i].reshape(-1, 1)
        n1= np.dot(w1, a0) + b1
        a1= f1(n1)

        n2= np.dot(w2, a1)+ b2
        a2= f2(n2)

        error= target- a2
        T_error+= error** 2

        s2 = (a2 - target) * df2(n2)
        s1 = np.dot(w2.T, s2) * df1(n1)

        w1 -= A* np.dot(s1, a0.T)
        b1 -= A* s1
        w2 -= A* np.dot(s2, a1.T)
        b2 -= A* s2
predictions= []
for i in range(len(data)):
    a0= data[i].reshape(-1, 1)
    n1= np.dot(w1, a0)+ b1
    a1 = f1(n1)

    n2 = np.dot(w2 ,a1) + b2
    a2 = f2(n2)
    predictions.append(1 if a2 >= 0.5 else 0)
predictions = np.array(predictions).reshape(len(data), 1)

plt.scatter(data[:, 0], data[:, 1], marker='o', c=predictions.flatten())
plt.show()
