import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X, Y, epotchs= 100, Alpha= 0.001):
    num_fitures= X.shape[1]
    w= np.zeros(num_fitures)
    b= 0
    for epoch in range(epotchs):
        E_flage= False
        for i in range(X.shape[0]):
            x_i= X[i]
            Y_i= Y[i]
            Z= np.dot(w, x_i)+ b
            Z= np.sign(Z)
            if Z != Y_i:
                w+= Alpha* Y_i* x_i
                b+= Alpha* Y_i
                E_flage= True
                if not E_flage:
                    break
    return w, b

#region make data
center1= 1
center2= 3
num_of_data= 200
row= np.ones(num_of_data).reshape(-1, 1)
numbers1= np.random.randn(num_of_data, 1)+ center1
dataset1= np.hstack((row, numbers1)) 
numbers2= np.random.randn(num_of_data, 1)+ center2
dataset2= np.hstack((row, numbers2))
lbl1= np.ones(num_of_data).reshape(-1, 1)
lbl2= -1 *  np.ones(num_of_data).reshape(-1, 1) 
X= np.vstack((dataset1, dataset2))
Y= np.vstack((lbl1, lbl2))  
#endregion

W, B= perceptron_train(X, Y)
o= np.array([-10, 10])
f = lambda o: W * o + B
d= f(o)

plt.plot(X[:200], Y[:200], "or")
plt.plot(X[200:], Y[200:], "ok")
plt.plot(o, d, "-g")
plt.show()
