import numpy as np
import math
import matplotlib.pyplot as plt

#region make fake data
number_of_data= 200
data1= np.random.randn(number_of_data, 2) + (0, 0)
data2= np.random.randn(number_of_data, 2) + (1, 3)
data3= np.random.randn(number_of_data, 2) + (4, 1)
data= np.vstack((data1, data2, data3))
#endregion

def d_until_o(centers, data):
    from_class= []
    for i in range(len(data)):
        dictancs= []
        for I in range(len(centers)):
            dx= data[i][0] - centers[I][0]
            dy= data[i][1] - centers[I][1]
            dt= math.sqrt((dx**2) + (dy**2))
            dictancs.append(dt)
        from_class.append(np.argmin(dictancs))
    from_class= np.array(from_class).reshape(-1, 1)
    dataset= np.hstack((data, from_class))
    return dataset
def updater_center(centers, dataset):
    new_centers= []
    for i in range(len(centers)):
        X= 0
        Y= 0
        count_menber_center= 0
        for j in range(len(dataset)):
            if dataset[j][2]== i:
                X+= dataset[j][0]
                Y+= dataset[j][1]
                count_menber_center += 1
        if count_menber_center > 0:
            X/= count_menber_center 
            Y/= count_menber_center
        C= (X, Y)
        new_centers.append(C)
    return np.array(new_centers)
def sho(centers, new_centers, data, dataset):
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], c= dataset[:, 2], marker= 'o')
    plt.plot(new_centers[:, 0], new_centers[:, 1], 'Xr')
    plt.pause(0.5)
#______________________________________________________________________________________________
centers= np.random.randn(3, 2)  # 3 primary centers

# print(centers)
# dataset= d_until_o(centers, data)
# print(dataset)
# print(np.shape(dataset))

new_centers= ([[10, 10], [10, 10], [10, 10]])
befor_centers= ([[2, 2], [2, 2], [2, 2]])

plt.plot(data[:, 0], data[:, 1], 'ok')
plt.show()
while not np.allclose(new_centers, befor_centers):
    dataset= d_until_o(centers, data)
    befor_centers= new_centers
    new_centers= updater_center(centers, dataset)
    centers= new_centers
    sho(centers, new_centers, data, dataset)

plt.show()
