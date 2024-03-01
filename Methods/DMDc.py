#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

from pydmd import DMDc
from sklearn.metrics import mean_squared_error
#from pydmd.plotter import plot_eigs

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from math import floor, ceil # python 3.x

import scipy.io

import csv

def writing_csv(path, data):
    with open(path, 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerows(data)


''' DATASET:
1 SRU dataset
    # matrix plot
    matrix_xlabel = 'Delayed state'; matrix_ylabel = 'Time (minute)'
    # array plot (when the plot is of one state)
    array_xlabel = 'Time (minute)'; array_ylabel = 'State'; array_title = 'State: x_' + str(column_to_show + 1)

2 Syntetic Complex dataset with complete U matrix (200 x 7160)
3 Syntetic Complex dataset with U[161,:] (fifth input with no delay)
    # matrix plot
    matrix_xlabel = 'State'; matrix_ylabel = 'Samples'
    # array plot (when the plot is of one state)
    array_xlabel = 'Samples'; array_ylabel = 'State'; array_title = 'State: x_' + str(column_to_show + 1)

4 V2G dataset with state NOT delayed and inputs NOT delayed
5 V2G datasets with state delayed and inputs NOT delayed
6 V2G datasets with state delayed and inputs delayed

dataset = 5
    XU1_DMDc 1 meteo + aggregated
    XU2_DMDc 1 aggregated
    XU3_DMDc 1 meteo (no rhum_t) + aggregated
    XU4_DMDc 1 meteo(no rhum_t)+aggregated(no holidays)
    XU5_DMDc 1 aggregated(no holidays)

dataset = 6 
    XU1_DMDc meteo + aggregated
    XU2_DMDc aggregated
    XU3_DMDc meteo (no rhum_t) + aggregated
    XU4_DMDc meteo(no rhum_t)+aggregated(no holidays)
    XU5_DMDc aggregated(no holidays)

    # matrix plot
    matrix_xlabel = 'Available Aggregated Capacity'; matrix_ylabel = 'Samples (30 minutes)'
    # array plot (when the plot is of one state)
    array_xlabel = 'Samples (30 minutes)'; array_ylabel = 'AAC'; array_title = 'State: ACC_' + str(column_to_show + 1)
'''

#insert path in wich to load .mat files
#load the training experimental file
path_for_load_experimental_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Train\XU5_DMDc.mat'
#load the test experimental file
path_for_load_experimental_test = r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Test\XU5test_DMDc 1.mat'

#insert path in which to save the csv files 
#save the training reconstructed file
path_for_save_reconstructed_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset reconstructed\Dataset DMDc\V2G\Stato ritardato ed ingressi ritardati\Train\XU5_DMDc'
#join ->    _reconstructed_train.csv
#save the test reconstructed file
path_for_save_reconstructed_test = r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset reconstructed\Dataset DMDc\V2G\Stato ritardato ed ingressi ritardati\Test\XU5_DMDc'   
#join ->    _reconstructed_test.csv                


#this parameter is used to decide which column to show
column_to_show = 0

''':param svd_rank: the rank for the truncation; If 0, the method computes the optimal rank and uses it for truncation;
 if positive interger, the method uses the argument for the truncation; if float between 0 and 1,the rank is the number 
 of the biggest singular values that are needed to reach the 'energy' specified by `svd_rank`; if -1, the method does
not compute truncation.'''
svd_rank_set = -1

# matrix plot
matrix_xlabel = 'Available Aggregated Capacity'
matrix_ylabel = 'Samples (30 minutes)'

# array plot (when the plot is of one state)
array_xlabel = 'Samples (30 minutes)'
array_ylabel = 'AAC'
array_title = 'State AAC_' + str(column_to_show + 1)


#dataset for traininig
D_mat = scipy.io.loadmat(path_for_load_experimental_train)
D_mat_list = [[element for element in upperElement] for upperElement in D_mat['X']]
U_mat_list = [[element for element in upperElement] for upperElement in D_mat['U']]
D_train = np.array(D_mat_list)
U_train = np.array(U_mat_list)


#dataset for testing
if path_for_load_experimental_test is not None:
    D_mat = scipy.io.loadmat(path_for_load_experimental_test)

    D_mat_list = [[element for element in upperElement] for upperElement in D_mat['X']]
    U_mat_list = [[element for element in upperElement] for upperElement in D_mat['U']]

    D_test = np.array(D_mat_list)
    U_test = np.array(U_mat_list)





'''from this moment on, nothing needs to be set'''

#if the data are an array 1-D with this instruction they became 2-D
if len(D_train.shape) == 1:
    D_train = D_train[: , np.newaxis].T
if len(U_train.shape) == 1:
    U_train = U_train[: , np.newaxis].T

vmax = np.amax(D_train)
vmin = np.amin(D_train)

#dataset for testing
if path_for_load_experimental_test is not None:
   
    #if the data are an array 1-D with this instruction they became 2-D
    if len(D_test.shape) == 1:
        D_test = D_test[:, np.newaxis].T
    if len(U_test.shape) == 1:
        U_test = U_test[:, np.newaxis].T




#eventually matrix D_train or U_train have different dimensions of columns (snapshots)
if D_train.shape[1] > U_train.shape[1]:
    D_train = D_train[:,:U_train.shape[1]]
else:
    U_train = U_train[:,:D_train.shape[1]]

#eventually matricies D_train or U_train have different dimensions of columns (snapshots) and different dimension from Test matricies
if path_for_load_experimental_test is not None:
    if D_test.shape[1] > U_test.shape[1]:
        D_test = D_test[:,:U_test.shape[1]]
        max = U_test.shape[1]
    else:
        U_test = U_test[:,:D_test.shape[1]]
        max = D_test.shape[1]

    D_train = D_train[:,:max]
    U_train = U_train[:,:max]



#number of rows of the dataset
x_train = np.linspace(0, D_train.shape[0], D_train.shape[0])

#number of columns of the dataset
t_train = np.linspace(0, D_train.shape[1], D_train.shape[1])


#this function allow to make plot like image (it is used to plot matrix values)
def make_plot(X, x=None, y=None, title='', xlabel = None, ylabel = None, vmin = None, vmax = None, ticks = None):
    """
    Plot of the data X
    """
    plt.figure()
    plt.title(title)
    if vmin is not None:
        CS = plt.pcolormesh(x, y, X, vmin = vmin, vmax = vmax, cmap= "viridis")
    else:
        plt.pcolor(X.real)
    plt.colorbar()
    if ticks is not None:
        plt.xticks(np.arange(0, len(X[0]), ticks))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


make_plot(D_train.T, x=x_train, y=t_train, title = 'Training dataset', xlabel = matrix_xlabel, ylabel = matrix_ylabel)


#plot of the state of the dataset selected
plt.figure()
plt.plot(t_train, D_train[column_to_show,:], 'g', label='Experimental data')
plt.title(array_title)
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.legend()
plt.show()




U_train = U_train[:,1:]

dmdc = DMDc(svd_rank = svd_rank_set)    

dmdc.fit(D_train,U_train)

DMDc_train_reconstructed = dmdc.reconstructed_data(control_input = None) 



#extraction of the matrix A calculation taken from the method reconstruction_data
eigs = np.power(dmdc.eigs, dmdc.dmd_time["dt"] // dmdc.original_time["dt"])
A = np.linalg.multi_dot([dmdc.modes, np.diag(eigs), np.linalg.pinv(dmdc.modes)])


writing_csv( path_for_save_reconstructed_train + '_A.csv' , A.real)
writing_csv(path_for_save_reconstructed_train + '_B.csv', dmdc.B.real)


plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.title('Experimental system')
plt.pcolor(D_train.real.T, vmin = vmin, vmax = vmax)
plt.colorbar()
plt.xlabel(matrix_xlabel)
plt.ylabel(matrix_ylabel)

plt.subplot(122)
plt.title('Reconstructed system')
plt.pcolor(DMDc_train_reconstructed.real.T, vmin = vmin, vmax = vmax)
plt.colorbar()
plt.xlabel(matrix_xlabel)
plt.ylabel(matrix_ylabel)

plt.show()






plt.figure()
plt.title(array_title)
plt.plot(t_train, D_train[column_to_show,:].real, 'b', label='Experimental data')
plt.plot(t_train, DMDc_train_reconstructed[column_to_show,:].real, 'g', label='DMDc reconstructed data')
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.legend()
plt.show()

plt.figure()
plt.title(array_title)
error=np.array(D_train[column_to_show,:].real) - np.array(DMDc_train_reconstructed[column_to_show,:].real)
plt.plot(t_train, error, 'b', label='Error')
plt.xlabel(array_xlabel)
plt.ylabel('Error')
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

def MSE(y_true, y_pred):
    mse_value = mean_squared_error(np.array(y_true.real), np.array(y_pred.real))
    return mse_value

def MAPE (y_true,y_pred):   #MEAN ABSOLUTE PERCENTAGE ERROR
    mape = mean_absolute_percentage_error(np.array(y_true.real), np.array(y_pred.real))
    return mape

def MAE(y_true, y_pred):     #MEAN ABSOLUTE ERROR
    mae_value = mean_absolute_error(np.array(y_true.real), np.array(y_pred.real))
    return mae_value

def RMSE(y_true, y_pred):     #ROOT MEAN SQUARED ERROR
    rmse_value = math.sqrt(MSE(y_true.real, y_pred.real))
    return rmse_value

def R2(y_true, y_pred):
    r2_value = r2_score(np.array(y_true.real), np.array(y_pred.real))
    return r2_value


print("Train KPI:")
print("MSE: ")
print((MSE(DMDc_train_reconstructed.T,D_train.T)))
print("{:.2e}".format(MSE(DMDc_train_reconstructed.T , D_train.T)))

print ("MAPE: ")
print (MAPE(DMDc_train_reconstructed.T , D_train.T))
print("{:.2e}".format(MAPE(DMDc_train_reconstructed.T , D_train.T)))

print ("MAE: ")
print(MAE(DMDc_train_reconstructed.T , D_train.T))
print("{:.2e}".format(MAE(DMDc_train_reconstructed.T , D_train.T)))

print ("RMSE: ")
print(RMSE(DMDc_train_reconstructed.T , D_train.T))
print("{:.2e}".format(RMSE(DMDc_train_reconstructed.T , D_train.T)))

print ("R2: ")
print(R2(DMDc_train_reconstructed.T , D_train.T))
print("{:.2e}".format(R2(DMDc_train_reconstructed.T , D_train.T)))





#show A and B of the model

plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.title('Matrix A')
plt.pcolor(A.real)
plt.colorbar()
plt.xlabel('State')
plt.ylabel('Output')

plt.subplot(122)
plt.title('Matrix B')
plt.pcolor(dmdc.B)
plt.colorbar()
plt.xticks(np.linspace(0, dmdc.B.shape[1], (U_train.shape[0] // D_train.shape[0]) + 1))
plt.xlabel('Input')
plt.ylabel('Output')

plt.show()







writing_csv(path_for_save_reconstructed_train + '_reconstructed_train.csv', DMDc_train_reconstructed.real)
writing_csv(path_for_save_reconstructed_train + '_experimental_train.csv' , D_train.real)




'''Test of the model'''
if path_for_load_experimental_test is not None:

    x_test = np.linspace(1, D_test.shape[0], D_test.shape[0])
    t_test = np.linspace(1, D_test.shape[1], D_test.shape[1])

    vmin = np.amax(D_test)
    vmax = np.amin(D_test)
    
    # D_test matrix plot
    make_plot(D_test.T, x=x_test, y=t_test, title = 'Test dataset', xlabel = matrix_xlabel, ylabel = matrix_ylabel)

    #plot of the state of the dataset selected
    plt.figure()
    plt.plot(t_test, D_test[column_to_show,:], 'g', label='Experimental data')
    plt.title(array_title)
    plt.xlabel(array_xlabel)
    plt.ylabel(array_ylabel)
    plt.legend()
    plt.show()

    U_test = U_test[:,1:]

    #reconstruction test
    DMDc_test_reconstructed = dmdc.reconstructed_data(control_input=U_test)


    # comparison between experimental test and reconstructed test
    plt.figure(figsize=(16, 6))

    plt.subplot(121)
    plt.title('Experimental system')
    plt.pcolor(D_test.real.T, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel(matrix_xlabel)
    plt.ylabel(matrix_ylabel)

    plt.subplot(122)
    plt.title('Reconstructed system')
    plt.pcolor(DMDc_test_reconstructed.real.T, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel(matrix_xlabel)
    plt.ylabel(matrix_ylabel)

    plt.show()



    plt.figure()
    plt.title(array_title)
    plt.plot(t_test, D_test[column_to_show,:], 'b', label='Experimental data')
    plt.plot(t_test, DMDc_test_reconstructed[column_to_show,:].real, 'g', label='DMDc reconstructed data')
    plt.xlabel(array_xlabel)
    plt.ylabel(array_ylabel)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(array_title)
    error=np.array(D_test[column_to_show,:]) - np.array(DMDc_test_reconstructed[column_to_show,:].real)
    plt.plot(t_test, error, 'b', label='Error')
    plt.xlabel(array_xlabel)
    plt.ylabel('Error')
    plt.legend()
    plt.show()


    print("Test KPI:")
    print("MSE: ")
    print((MSE(DMDc_test_reconstructed.T,D_test.T)))
    print("{:.2e}".format(MSE(DMDc_test_reconstructed.T , D_test.T)))

    print ("MAPE: ")
    print (MAPE(DMDc_test_reconstructed.T , D_test.T),"%")
    print("{:.2e}".format(MAPE(DMDc_test_reconstructed.T , D_test.T)))

    print ("MAE: ")
    print(MAE(DMDc_test_reconstructed.T , D_test.T))
    print("{:.2e}".format(MAE(DMDc_test_reconstructed.T , D_test.T)))

    print ("RMSE: ")
    print(RMSE(DMDc_test_reconstructed.T , D_test.T))
    print("{:.2e}".format(RMSE(DMDc_test_reconstructed.T , D_test.T)))

    print ("R2: ")
    print(R2(DMDc_test_reconstructed.T , D_test.T))
    print("{:.2e}".format(R2(DMDc_test_reconstructed.T , D_test.T)))


    writing_csv(path_for_save_reconstructed_test + '_reconstructed_test.csv', DMDc_test_reconstructed.real)
    writing_csv(path_for_save_reconstructed_test + '_experimental_test.csv', D_test.real)









