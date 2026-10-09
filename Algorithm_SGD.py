import time
import numpy as np
import matplotlib.pyplot as plt

#region creat fake data
x_data= np.linspace(50, 200, 90)
fake_fanction= lambda x: 4 * x + 20
y_data= fake_fanction(x_data)
noize= np.random.randn(90)*5
y_data += noize 
#endregion

#create random weights
w1= np.random.randn(1)
w2= np.random.randn(1)

#region fonction
def error_fonction(x_data, y_data, w1, w2):
  # error function (mean squared error)
    sum= 0
    for i in range(len(x_data)):
        sum += (y_data[i]- (w1* x_data[i]+ w2))** 2
    error=  sum / 2* len(x_data)
    return error

def updator_w1(x_data, y_data, w1, w2, Alpha= 0.0001):
    sum= 0
    for i in range(len(x_data)):
      # The partial derivative of the error with respect to w1
        sum += (-1* x_data[i]) * (y_data[i]- (w1* x_data[i]+ w2))
    
    # update w1
    w1 = w1- Alpha* sum/ len(x_data)
    return w1

def updator_w2(x_data, y_data, w1, w2, Alpha=  0.0001):
    sum= 0
    for i in range(len(x_data)):
      # The partial derivative of the error with respect to w2
        sum += (-1)* (y_data[i]- (w1* x_data[i]+ w2))
    
    # update w2
    w2= w2- Alpha* sum/ len(x_data)
    return w2

def sho(x_data, y_data, w1, w2):
  # Graphical display function
    f= lambda x: w1* x+ w2
    y= f(x_data)
    plt.clf()
    plt.plot(x_data, y_data, "or")
    plt.plot(x_data, y, "-b")
    plt.pause(0.1)

#endregion

# setting error for entering the loop  
befor_error= -1000
now_error= 1000

#loop for learning & animation display
while abs(now_error - befor_error) >  0.0001:
    """The learning stopping condition when the difference between
          the current error and the previous error becomes less than 0.0001"""
    befor_error= error_fonction(x_data, y_data, w1, w2)
    w1= updator_w1(x_data, y_data, w1, w2)
    w2= updator_w2(x_data, y_data, w1, w2)
    now_error= error_fonction(x_data, y_data, w1, w2)
    sho(x_data, y_data, w1, w2)
    time.sleep(1)

plt.show()
