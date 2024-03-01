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
path_for_load_experimental_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Train\XU1_DMDc.mat'
#load the test experimental file
path_for_load_experimental_test = r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Test\XU1test_DMDc 1.mat'

#insert path in which to save the csv files 
#save the training reconstructed file
path_for_save_reconstructed_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset reconstructed\Dataset DMDc\V2G\Stato ritardato ed ingressi ritardati\Train\XU1_DMDc_reconstructed_train_meteo + aggregated.csv'
#join ->    _reconstructed_train.csv
#save the test reconstructed file
path_for_save_reconstructed_test = r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset reconstructed\Dataset DMDc\V2G\Stato ritardato ed ingressi ritardati\Test\XU1_DMDc_reconstructed_test_meteo + aggregated.csv'   
#join ->    _reconstructed_test.csv                


#this parameter is used to decide which column to show
column_to_show = 0

''':param svd_rank: the rank for the truncation; If 0, the method computes the optimal rank and uses it for truncation;
 if positive interger, the method uses the argument for the truncation; if float between 0 and 1,the rank is the number 
 of the biggest singular values that are needed to reach the 'energy' specified by `svd_rank`; if -1, the method does
not compute truncation.'''
svd_rank_set = -1


#sampling time 
sampling_time_hour = 0.5

#prediction step in hour
prediction_k_hour = 24

#prediction step in samples
prediction_k = math.floor(prediction_k_hour // sampling_time_hour)

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

DMDc_train_reconstructed = []

prediction_slots = math.floor(D_train.shape[1] // prediction_k)

for k in range(0, prediction_slots):
    D_slot = D_train[: , k * prediction_k:]
    U_slot = U_train[: , k * prediction_k:]
    if k == 185:
        debug = 0
    zero_padding_lenght = int(U_train.shape[1] - U_slot.shape[1])
    zeros_matrix_D = np.zeros([D_train.shape[0], zero_padding_lenght])
    zeros_matrix_U = np.zeros([U_train.shape[0], zero_padding_lenght])
    D_slot = np.hstack([D_slot, zeros_matrix_D])
    U_slot = np.hstack([U_slot, zeros_matrix_U])
    reconstructed_slot = dmdc.reconstructed_data(open_loop = False, X = D_slot, control_input = U_slot)[:,:prediction_k].real 
    DMDc_train_reconstructed.append(reconstructed_slot)

DMDc_train_reconstructed = np.hstack(DMDc_train_reconstructed)

D_train_truncated = D_train[:,:DMDc_train_reconstructed.shape[1]]

#extraction of the matrix A calculation taken from the method reconstruction_data
eigs = np.power(dmdc.eigs, dmdc.dmd_time["dt"] // dmdc.original_time["dt"])
A = np.linalg.multi_dot([dmdc.modes, np.diag(eigs), np.linalg.pinv(dmdc.modes)])





plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.title('Experimental system')
plt.pcolor(D_train_truncated.real.T, vmin = vmin, vmax = vmax)
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


t_train = t_train[:DMDc_train_reconstructed.shape[1]]



plt.figure()
plt.title(array_title)
plt.plot(t_train, D_train_truncated[column_to_show,:].real, 'b', label='Experimental data')
plt.plot(t_train, DMDc_train_reconstructed[column_to_show,:].real, 'g', label='DMDc reconstructed data')
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.legend()
plt.show()

plt.figure()
plt.title(array_title)
error=np.array(D_train_truncated[column_to_show,:].real) - np.array(DMDc_train_reconstructed[column_to_show,:].real)
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
print((MSE(DMDc_train_reconstructed.T,D_train_truncated.real.T)))
print("{:.2e}".format(MSE(DMDc_train_reconstructed.T , D_train_truncated.real.T)))

print ("MAPE: ")
print (MAPE(DMDc_train_reconstructed.T , D_train_truncated.real.T))
print("{:.2e}".format(MAPE(DMDc_train_reconstructed.T , D_train_truncated.real.T)))

print ("MAE: ")
print(MAE(DMDc_train_reconstructed.T , D_train_truncated.real.T))
print("{:.2e}".format(MAE(DMDc_train_reconstructed.T , D_train_truncated.real.T)))

print ("RMSE: ")
print(RMSE(DMDc_train_reconstructed.T , D_train_truncated.real.T))
print("{:.2e}".format(RMSE(DMDc_train_reconstructed.T , D_train_truncated.real.T)))

print ("R2: ")
print(R2(DMDc_train_reconstructed.T , D_train_truncated.real.T))
print("{:.2e}".format(R2(DMDc_train_reconstructed.T , D_train_truncated.real.T)))





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





def writing_csv(path, data):
    with open(path, 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerows(data)


writing_csv(path_for_save_reconstructed_train, DMDc_train_reconstructed.real)




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


    DMDc_test_reconstructed = []

    prediction_slots = math.floor(D_test.shape[1] // prediction_k)

    #reconstruction test
    for k in range(0, prediction_slots):
        D_slot = D_test[: , k * prediction_k:]
        U_slot = U_test[: , k * prediction_k:]
        if k == 185:
            debug = 0
        zero_padding_lenght = int(U_test.shape[1] - U_slot.shape[1])
        zeros_matrix_D = np.zeros([D_test.shape[0], zero_padding_lenght])
        zeros_matrix_U = np.zeros([U_test.shape[0], zero_padding_lenght])
        D_slot = np.hstack([D_slot, zeros_matrix_D])
        U_slot = np.hstack([U_slot, zeros_matrix_U])
        reconstructed_slot = dmdc.reconstructed_data(open_loop = False, X = D_slot, control_input = U_slot).real 
        if k == 0 or k == 1:
            t = np.linspace(0, reconstructed_slot.shape[1], reconstructed_slot.shape[1])
            plt.figure()
            plt.plot(t, reconstructed_slot[column_to_show , :], label = 'reconstructed_slot')
            plt.plot(t, D_slot[column_to_show, :], label = 'D_slot')
            plt.plot(t, reconstructed_slot[24 , :], label = 'reconstructed_slot_24')
            plt.plot(t, D_slot[24, :], label = 'D_slot_24')
            plt.plot(t, reconstructed_slot[47 , :], label = 'reconstructed_slot_48')
            plt.plot(t, D_slot[47, :], label = 'D_slot_48')
            plt.legend()
            plt.show()
        reconstructed_slot = reconstructed_slot[:,:prediction_k]
        DMDc_test_reconstructed.append(reconstructed_slot)

    DMDc_test_reconstructed = np.hstack(DMDc_test_reconstructed)

    D_test_truncated = D_test[:,:DMDc_test_reconstructed.shape[1]]


    # comparison between experimental test and reconstructed test
    plt.figure(figsize=(16, 6))

    plt.subplot(121)
    plt.title('Experimental system')
    plt.pcolor(D_test_truncated.real.T, vmin = vmin, vmax = vmax)
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

    t_test = t_test[:DMDc_test_reconstructed.shape[1]]


    plt.figure()
    plt.title(array_title)
    plt.plot(t_test, D_test_truncated[column_to_show,:], 'b', label='Experimental data')
    plt.plot(t_test, DMDc_test_reconstructed[column_to_show,:].real, 'g', label='DMDc reconstructed data')
    plt.xlabel(array_xlabel)
    plt.ylabel(array_ylabel)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(array_title)
    error=np.array(D_test_truncated[column_to_show,:]) - np.array(DMDc_test_reconstructed[column_to_show,:].real)
    plt.plot(t_test, error, 'b', label='Error')
    plt.xlabel(array_xlabel)
    plt.ylabel('Error')
    plt.legend()
    plt.show()


    print("Test KPI:")
    print("MSE: ")
    print((MSE(DMDc_test_reconstructed.T,D_test_truncated.T)))
    print("{:.2e}".format(MSE(DMDc_test_reconstructed.T , D_test_truncated.T)))

    print ("MAPE: ")
    print (MAPE(DMDc_test_reconstructed.T , D_test_truncated.T),"%")
    print("{:.2e}".format(MAPE(DMDc_test_reconstructed.T , D_test_truncated.T)))

    print ("MAE: ")
    print(MAE(DMDc_test_reconstructed.T , D_test_truncated.T))
    print("{:.2e}".format(MAE(DMDc_test_reconstructed.T , D_test_truncated.T)))

    print ("RMSE: ")
    print(RMSE(DMDc_test_reconstructed.T , D_test_truncated.T))
    print("{:.2e}".format(RMSE(DMDc_test_reconstructed.T , D_test_truncated.T)))

    print ("R2: ")
    print(R2(DMDc_test_reconstructed.T , D_test_truncated.T))
    print("{:.2e}".format(R2(DMDc_test_reconstructed.T , D_test_truncated.T)))


    writing_csv(path_for_save_reconstructed_test, DMDc_test_reconstructed.real)













def reconstructed_data(self, open_loop, X, control_input=None):
    """
    Return the reconstructed data, computed using the `control_input`
    argument. If the `control_input` is not passed, the original input (in
    the `fit` method) is used. The input dimension has to be consistent
    with the dynamics.

    :param numpy.ndarray control_input: the input control matrix, open_loop = True if we use original data at step k to calculate the output at state k+1
    :return: the matrix that contains the reconstructed snapshots.
    :rtype: numpy.ndarray
    """
    controlin = (
        np.asarray(control_input)
        if control_input is not None
        else self._controlin
    )

    if controlin.shape[-1] != self.dynamics.shape[-1] - 1:
        raise RuntimeError(
            "The number of control inputs and the number of snapshots to "
            "reconstruct has to be the same"
        )

    eigs = np.power(
        self.eigs, self.dmd_time["dt"] // self.original_time["dt"]
    )
    A = np.linalg.multi_dot(
        [self.modes, np.diag(eigs), np.linalg.pinv(self.modes)]    
    )

    if X is None:
        X = self.snapshots
        data = [self.snapshots[:, 0]]
    else:
        data = [X[:,0]]
    expected_shape = data[0].shape

    for i, u in enumerate(controlin.T):
        #open loop reconstruction 
        if open_loop == True:
            arr = A.dot(X[:,i]) + self._B.dot(u)
        else:
            arr = A.dot(data[i]) + self._B.dot(u)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Invalid shape: expected {expected_shape}, got {arr.shape}"
            )
        data.append(arr)

    data = np.array(data).T

    #return data

    return data
