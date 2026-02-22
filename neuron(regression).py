import matplotlib.pyplot as plt
import numpy as np

#region maked fake data
num_of_data= 80
fake_x= np.linspace(-5, 2, num_of_data)
g= lambda x : 1+ np.cos(np.pi/7* x)
fake_y= g(fake_x)
#endregion

f1= lambda x: 1/ (1+ np.exp(-x)) # log_sigmoid
df1= lambda x: f1(x)* (1- f1(x))
f2= lambda x: x  #Linear
df2= 1

w1= np.array([0.2, 0.3]).reshape(2, 1)
b1= np.array([0.5, 0.5]).reshape(2, 1)
w2= np.array([0.25, 0.35]).reshape(1, 2)
b2= np.array([0.4]).reshape(1, 1)
A= 0.1

for epoc in range(1000):    
    for i in range(len(fake_x)):
        a0= fake_x[i]
        n1= w1 * a0 + b1
        a1= f1(n1)

        n2= np.dot(w2, a1)+ b2
        a2= f2(n2)

        target= g(a0)
        error= target- a2

        s2= (a2 - target)*2
        fp1= np.array([[df1(n1[0])[0],0],[0,df1(n1[1])[0]]])
        s1= np.dot(fp1, w2.T)* s2

        w1 -= A* np.dot(s1, a0)
        b1 -= A* s1
        w2 -= A* np.dot(s2, a1.T)
        b2 -= A* s2
predictions= []
for i in range(len(fake_x)):
    a0= fake_x[i]
    n1= w1* a0+ b1
    a1 = f1(n1)

    n2 = np.dot(w2 ,a1) + b2
    a2 = f2(n2)
    predictions.append(a2)
predictions = np.array(predictions).reshape(num_of_data,1)

plt.scatter(fake_x, fake_y, marker='s', c='m')
plt.scatter(fake_x, predictions, marker='o', c='g')
plt.legend(["Orginal data", "prediction"])
plt.show()
