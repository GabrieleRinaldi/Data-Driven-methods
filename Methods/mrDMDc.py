import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from math import floor, ceil # python 3.x

import scipy.io

import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

from sklearn.metrics import mean_squared_error
#from pydmd.plotter import plot_eigs

import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from math import floor, ceil # python 3.x

import scipy.io

#from pydmd import dmdc


"""
Derived module from dmdbase.py for dmd with control.

Reference:
- Proctor, J.L., Brunton, S.L. and Kutz, J.N., 2016. Dynamic mode decomposition
with control. SIAM Journal on Applied Dynamical Systems, 15(1), pp.142-161.
"""
import numpy as np

from pydmd.dmdbase import DMDBase
from pydmd.dmdoperator import DMDOperator
from pydmd.snapshots import Snapshots
from pydmd.utils import compute_svd, compute_tlsq
from pydmd.plotter import plot_eigs_mrdmd


import control as ct
from scipy import signal

import harold


class DMDControlOperator(DMDOperator):
    """
    DMD with control base operator. This should be subclassed in order to
    implement the appropriate features.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param svd_rank_omega: the rank for the truncation of the aumented matrix
        omega composed by the left snapshots matrix and the control. Used only
        for the `_fit_B_unknown` method of this class. It should be greater or
        equal than `svd_rank`. For the possible values please refer to the
        `svd_rank` parameter description above.
    :type svd_rank_omega: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    """

    def __init__(self, svd_rank, svd_rank_omega, tlsq_rank):
        super(DMDControlOperator, self).__init__(
            svd_rank=svd_rank,
            exact=True,
            rescale_mode=None,
            forward_backward=False,
            sorted_eigs=False,
            tikhonov_regularization=None,
        )
        self._svd_rank_omega = svd_rank_omega
        self._tlsq_rank = tlsq_rank


class DMDBKnownOperator(DMDControlOperator):
    """
    DMD with control base operator when B is given.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param svd_rank_omega: the rank for the truncation of the aumented matrix
        omega composed by the left snapshots matrix and the control. Used only
        for the `_fit_B_unknown` method of this class. It should be greater or
        equal than `svd_rank`. For the possible values please refer to the
        `svd_rank` parameter description above.
    :type svd_rank_omega: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    """

    def compute_operator(self, X, Y, B, controlin):
        """
        Compute the low-rank operator. This is the standard version of the DMD
        operator, with a correction which depends on B.

        :param numpy.ndarray X: matrix containing the snapshots x0,..x{n-1} by
            column.
        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :param numpy.ndarray B: the matrix B.
        :param numpy.ndarray control: the control input.
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, the (truncated) right-singular vectors
            matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        X, Y = compute_tlsq(X, Y, self._tlsq_rank)
        Y = Y - B.dot(controlin)
        return super(DMDBKnownOperator, self).compute_operator(X, Y)


class DMDBUnknownOperator(DMDControlOperator):
    """
    DMD with control base operator when B is unknown.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param svd_rank_omega: the rank for the truncation of the aumented matrix
        omega composed by the left snapshots matrix and the control. Used only
        for the `_fit_B_unknown` method of this class. It should be greater or
        equal than `svd_rank`. For the possible values please refer to the
        `svd_rank` parameter description above.
    :type svd_rank_omega: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    """

    def max_value(matrix):
        max_value = np.max([np.max(matrix)])
        min_value = np.min([np.max(matrix)])
        if (max_value >= -(min_value)):
            scale = max_value
        else:
            scale = -min_value
        return scale


    def compute_operator(self, X, Y, controlin):
        """
        Compute the low-rank operator.

        :param numpy.ndarray X: matrix containing the snapshots x0,..x{n-1} by
            column.
        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :param numpy.ndarray control: the control input.
        :return: the (truncated) left-singular vectors matrix of Y, and
            the product between the left-singular vectors of Y and Btilde.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        snapshots_rows = X.shape[0]

        omega = np.vstack([X, controlin])

        Up, sp, Vp = compute_svd(omega, self._svd_rank_omega)

        Up1 = Up[:snapshots_rows, :]
        Up2 = Up[snapshots_rows:, :]

        Ur, _, _ = compute_svd(Y, self._svd_rank)

        self._Atilde = np.linalg.multi_dot(
            [Ur.T.conj(), Y, Vp, np.diag(np.reciprocal(sp)), Up1.T.conj(), Ur]
        )

        self._compute_eigenquantities()
        self._compute_modes(Y, sp, Vp, Up1, Ur)

        Btilde = np.linalg.multi_dot(
            [Ur.T.conj(), Y, Vp, np.diag(np.reciprocal(sp)), Up2.T.conj()] 
        )

        return Ur, Ur.dot(Btilde)

    def _compute_modes(self, Y, sp, Vp, Up1, Ur):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator (stored in self.modes and self.Lambda).
        """

        self._modes = np.linalg.multi_dot(
            [
                Y,
                Vp,
                np.diag(np.reciprocal(sp)),
                Up1.T.conj(),
                Ur,
                self.eigenvectors,
            ]
        )
        self._Lambda = self.eigenvalues


class DMDc(DMDBase):
    """
    Dynamic Mode Decomposition with control.
    This version does not allow to manipulate the temporal window within the
    system is reconstructed.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param opt: argument to control the computation of DMD modes amplitudes.
        See :class:`DMDBase`. Default is False.
    :type opt: bool or int
    :param svd_rank_omega: the rank for the truncation of the aumented matrix
        omega composed by the left snapshots matrix and the control. Used only
        for the `_fit_B_unknown` method of this class. It should be greater or
        equal than `svd_rank`. For the possible values please refer to the
        `svd_rank` parameter description above.
    :type svd_rank_omega: int or float
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, opt=False, svd_rank_omega=-1):
        # we're going to initialize Atilde when we know if B is known
        self._Atilde = None
        # remember the arguments for when we'll need them
        self._dmd_operator_kwargs = {
            "svd_rank": svd_rank,
            "svd_rank_omega": svd_rank_omega,
            "tlsq_rank": tlsq_rank,
        }

        self._opt = opt
        self._exact = False

        self._B = None
        self._snapshots_holder = None
        self._controlin = None
        self._basis = None

        self._modes_activation_bitmask_proxy = None

        self.level = None            # level of recursion
        self.bin_num = None        # time bin number
        self.bin_size = None      # time bin size
        self.start = None          # starting index
        self.stop = None # stopping index
        self.step = None              # step size
        self.dato = None
        self.nyq = None
        self.rho = None

        self.A = None

        self.D_dmdc_A = None
        self.D_dmdc_B = None

        self.mu_SLOW_step = None
        self.phi_SLOW_step = None

        self.sys_cont = None
        self.mu_SLOW_cont = None

        self.sys_disc_1 = None
        self.mu_SLOW_1 = None

        self.percentage_of_filtration = None





    @property
    def svd_rank_omega(self):
        return self.operator._svd_rank_omega

    @property
    def B(self):
        """
        Get the operator B.

        :return: the operator B.
        :rtype: numpy.ndarray
        """
        return self._B

    @property
    def basis(self):
        """
        Get the basis used to reduce the linear operator to the low dimensional
        space.

        :return: the matrix which columns are the basis vectors.
        :rtype: numpy.ndarray
        """
        return self._basis


    def max_value(matrix):
        max_value = np.max([np.max(matrix)])
        min_value = np.min([np.max(matrix)])
        if (max_value >= -(min_value)):
            scale = max_value
        else:
            scale = -min_value
        return scale



    '''GABRIELE aggiunti parametri eigs e modes per override dopo la filtrazione degli SLOW'''

    '''
    def reconstructed_data(self, control_input=None, eigs = None, modes = None, _B = None):
        """
        Return the reconstructed data, computed using the `control_input`
        argument. If the `control_input` is not passed, the original input (in
        the `fit` method) is used. The input dimension has to be consistent
        with the dynamics.

        :param numpy.ndarray control_input: the input control matrix.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        controlin = (
            np.asarray(control_input)
            if control_input is not None
            else self._controlin
        )

        if controlin.shape[-1] != self.dynamics.shape[-1] - 1:                  #controlin 200x8
            raise RuntimeError(
                "The number of control inputs and the number of snapshots to "
                "reconstruct has to be the same"
            )
                                                                                #bin_size / step
        
        #GABRIELE imposizione degli autovalori dopo filtrazione di SLOW
        if eigs is None:
            eigs = np.power(                                                         #eigs vettore di dimensione 8 per livello 0
                self.eigs, self.dmd_time["dt"] // self.original_time["dt"]
            )

        #GABRIELE imposizione degI modes dopo filtrazione di SLOW
        if modes is None:
            modes = self.modes

        if _B is None:
            _B = self._B
        
        A = np.linalg.multi_dot(                                                  #A è una matrice 40x40 per livello 0
            [modes, np.diag(eigs), np.linalg.pinv(modes)]    
        )

        A = A.real
        self.A = A

        x = np.linspace(0, A.shape[0], A.shape[0])
        y = np.linspace(0, A.shape[1], A.shape[1])
        
        max_value = np.max([np.max(A)])
        min_value = np.min([np.min(A)])
        if (max_value >= -(min_value)):
            scale = max_value
        else:
            scale = -min_value
        
        
        #make_plot(A.real, x=y, y=x, title = 'A ' + str(self.level), xlabel = 'Input', ylabel = 'Output', vmin = -(scale), vmax = scale)

        data = [self.snapshots[:, 0]]
        expected_shape = data[0].shape


        #result = dot(self.dynamics,modes) + _B.dot(self._controlin)


        #data[i] rappresenta l'istante i-esimo dei dati dello stato
        #u rappresenta l'istante i-esimo dei dati di input
        for i, u in enumerate(controlin.T):
            arr = A.dot(data[i]) + _B.dot(u)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Invalid shape: expected {expected_shape}, got {arr.shape}"
                )
            data.append(arr)

        data = np.array(data).T

        return data
        '''




    def reconstructed_data(self, control_input=None):
        """
        Return the reconstructed data, computed using the `control_input`
        argument. If the `control_input` is not passed, the original input (in
        the `fit` method) is used. The input dimension has to be consistent
        with the dynamics.

        :param numpy.ndarray control_input: the input control matrix.
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

        data = [self.snapshots[:, 0]]
        expected_shape = data[0].shape

        for i, u in enumerate(controlin.T):
            arr = A.dot(data[i]) + self._B.dot(u)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Invalid shape: expected {expected_shape}, got {arr.shape}"
                )
            data.append(arr)

        data = np.array(data).T

        return data


    


    def fit(self, X, I, B=None):
        """
        Compute the Dynamic Modes Decomposition with control given the original
        snapshots and the control input data. The matrix `B` that controls how
        the control input influences the system evolution can be provided by
        the user; otherwise, it is computed by the algorithm.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param I: the control input.
        :type I: numpy.ndarray or iterable
        :param numpy.ndarray B: matrix that controls the control input
            influences the system evolution.
        :type B: numpy.ndarray or iterable
        """
        self._reset()

        self._snapshots_holder = Snapshots(X)
        self._controlin = np.atleast_2d(np.asarray(I))

        n_samples = self.snapshots.shape[1]
        X = self.snapshots[:, :-1]
        Y = self.snapshots[:, 1:]

        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )
        

        if B is None:
            self._Atilde = DMDBUnknownOperator(**self._dmd_operator_kwargs)
            self._basis, self._B = self.operator.compute_operator(
                X, Y, self._controlin
            )
        else:
            self._Atilde = DMDBKnownOperator(**self._dmd_operator_kwargs)
            U, _, _ = self.operator.compute_operator(X, Y, B, self._controlin)

            self._basis = U
            self._B = B

        self._b = self._compute_amplitudes()

        return self



import csv
#export of data to csv file
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


#this parameter is used to decide which column to show
column_to_show = 0

# matrix plot
matrix_xlabel = 'Available Aggregated Capacity'
matrix_ylabel = 'Samples (30 minutes)'

# array plot (when the plot is of one state)
array_xlabel = 'Samples(30 minutes)'
array_ylabel = 'ACC'
array_title = 'ACC x_' + str(column_to_show + 1)


#insert path in wich to load .mat files
#load the training experimental file
path_for_load_experimental_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Train\XU5_DMDc.mat'
#load the test experimental file
path_for_load_experimental_test = None#r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset experimental\V2G\Stato ritardato ed ingressi ritardati\Test\XU5test_DMDc 1.mat'

#insert path in which to save the csv files 
#save the training reconstructed file
path_for_save_reconstructed_train =  r'C:\Users\gabri\Desktop\Università\Tirocinio\Dataset reconstructed\Dataset mrDMDc\V2G\Stato ritardato ed ingressi ritardati\Train\XU5_mrDMDc'  #skip suffix '.csv'
#save the test reconstructed file
path_for_save_reconstructed_test = None#'C:\\Users\\gabri\\Desktop\\Università\\Tirocinio\\Dataset reconstructed\\Dataset DMDc\\V2G\\Stati ritardati ed ingressi ritardati\\Test\\XU1_DMDc_reconstructed_test_meteo + aggregated.csv'                   


#svd_rank for DMDc object
''':param svd_rank: the rank for the truncation; If 0, the method computes the optimal rank and uses it for truncation;
 if positive interger, the method uses the argument for the truncation; if float between 0 and 1,the rank is the number 
 of the biggest singular values that are needed to reach the 'energy' specified by `svd_rank`; if -1, the method does
not compute truncation.'''
svd_rank_set = -1


#max_cycles for mrDMDc
max_cycles_set = 10
'''
there is a possibility that if the reconstruction is not good (an example if the reconstructed system in unstable), 
changing the parameter will work. 
No way has been figured out to decide whether to increase or decrease it in the event of a bad reconstruction
 '''

#index of filtration, needs to set the size of rho
slow_feature_scale = 5


'''selection of the reconstruction type
1 Reconstruction with A and B dt = step
2 Reconstruction with A and B dt = 1 (trasformation of A and B with Harold library)
3 Reconstruction with forced response (work in progress)
'''
reconstruction = 1








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
    #plt.figure()
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
    #plt.show()


make_plot(D_train.T, x=x_train, y=t_train, title = 'Training dataset', xlabel = matrix_xlabel, ylabel = matrix_ylabel)


#plot of the state of the dataset selected
plt.figure()
plt.plot(t_train, D_train[column_to_show,:], 'g', label='Experimental data')
plt.title(array_title)
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.legend()
plt.show()




#this function make a comparison between the first column of the original system and the reconstructed system
def comparison(D, D_mrdmdc_level, level, title=''):
        D0 = D[column_to_show,:]
        D_0 = D_mrdmdc_level[column_to_show,:]

        t_train = D0.shape[0]
        t_train = np.linspace(0, t_train, t_train)
    
        #plt.figure()
        plt.plot(t_train, D0.real, 'b', label='Experimental data')
        plt.plot(t_train, D_0.real, 'g', label='mrDMDc level: ' + str(level))
        plt.xlabel(array_xlabel)
        plt.ylabel(array_ylabel)
        plt.title(array_title + ' ' + title)
        plt.legend()
        #plt.show()












def mrdmdc(D, U, level=0, bin_num=0, offset=0, max_levels=20, max_cycles=max_cycles_set):
    """Compute the multi-resolution DMDc on the dataset `D`, returning a list of nodes
    in the hierarchy. Each node represents a particular "time bin" (window in time) at
    a particular "level" of the recursion (time scale). The node is an object consisting
    of the various data structures generated by the DMDc at its corresponding level and
    time bin. The `level`, `bin_num`, and `offset` parameters are for record keeping 
    during the recursion and should not be modified.
    The `max_levels` parameter controls the maximum number of levels. The `max_cycles`
    parameter controls the maximum number of mode oscillations in any given time scale 
    that qualify as 'slow'."""
     
    
    # 4 times nyquist limit to capture cycles                 
    nyq = 4 * max_cycles                                      

    
    #bin_size is equal to number of columns of D
    bin_size = D.shape[1]                          


    #the condition for exit from algorithm
    if (bin_size) < (nyq):                                        
        return []


    # extract subsamples, take a value every step for D and U
    step = bin_size // nyq        
    D_sub_step = D[:,::(step)]                                           
    U_sub_step = U[:,::(step)]            





    

    #declaration of DMDc object, it will be used to calculate eigenvalues and modes, 
    #also keeps track of the algorithm iterations and allow to save the reconstructed matrix for each iteration
    dmdc = DMDc(svd_rank = svd_rank_set) 

    #save the DMDc object into an array
    nodes = [dmdc]

    dmdc.level = level            # level of recursion
    dmdc.bin_num = bin_num        # time bin number
    dmdc.bin_size = bin_size      # time bin size
    dmdc.start = offset           # starting index
    dmdc.stop = offset + bin_size # stopping index
    dmdc.step = step              # step size
    dmdc.nyq = nyq


    #fitting model from data and input passed to algorithm, take the snapshot at k+1
    dmdc.fit(D_sub_step , U_sub_step[:,1:])


    #extract eigenvalues and modes from DMDc object
    mu = dmdc.eigs
    Phi = dmdc.modes
    

    #Slow filtration
    rho = slow_feature_scale*max_cycles / bin_size                               
    dmdc.rho = rho


    # consolidate slow eigenvalues (as boolean mask)
    slow = (np.abs(np.log(mu) / (2 * pi * step))) <= rho
        

    # number of slow modes
    n = sum(slow)                        


    #calculate the percentage of filtration 100 : mu = percentage_of_filtration : n
    dmdc.percentage_of_filtration = math.ceil(100 - ((100/len(mu)) * n))


    # extract slow modes (perhaps empty)                       
    mu_SLOW_step = mu[slow]                                               
    Phi_SLOW_step = Phi[:,slow]                                        
    

    #if found slow mode...
    if n > 0:                                                  

        '''Definition of StateSpace discrete system with dt = step'''
        #Calculate A with slow features, they were calcuted with dt = step 
        A_disc_step = np.linalg.multi_dot(                                                  
            [Phi_SLOW_step, np.diag(mu_SLOW_step), np.linalg.pinv(Phi_SLOW_step)]    
        )

        #Extract B from dmdc object, also calculate with dt = step by method .fit (when we call dmdc.fit)         
        B_disc_step = dmdc.B     

        #Considerate two matrices C and D
        #matrix C with a ones column and the rest with 0 to extract the first state (that is not delayed) 
        C_disc = np.zeros([A_disc_step.shape[0],B_disc_step.shape[0]], dtype = 'complex')
        C_disc[:,0] = 1

        #matrix D with 0 because the inputs don't influence directly the exit (strictly own system)
        D_disc = np.zeros([A_disc_step.shape[0], B_disc_step.shape[1]], dtype = 'complex')            

        #declaration of StateSpace system, helps to convert discrete system with dt = step in continuous and see eigenvalues in continuous
        sys_disc_step = harold.State(A_disc_step, B_disc_step , C_disc, D_disc, dt = (step))

       
        '''Defintion of StateSpace continuous system'''
        #to see eigenvalues in continuous used the method of the Harold library "undiscretize" 
        #that allow to pass from dicrete system to continuous system 
        sys_cont = harold.undiscretize(sys_disc_step, method='tustin')  

        #extract matrices from continuous system
        A_cont = sys_cont.a
        B_cont = sys_cont.b
        C_cont = sys_cont.c
        D_cont = sys_cont.d
        
        #calculate eigenvalues from A matrix
        [dmdc.mu_SLOW_cont,eigenvectors] = eig(A_cont)


        #to print pzmap (pole-zero map) about system, 
        #there is the necessity to use another library because Harold library don't support the pzmap method
        #the library that allow to see pzmap is the "Control Library for Python" doc: https://python-control.readthedocs.io/en/0.9.4/index.html
        sys_cont_control = ct.StateSpace(A_cont, B_cont, C_cont, D_cont)    
        #ct.pzmap(sys_cont_control, title= 'Pole Zero Map node: ' + str(number_node) + ' level: ' + str(level) + ' step: ' + str(step))
        #plt.show()

        
        '''Definition of StateSpace discrete system with dt = 1'''
        #for the reconstruction there is a necessity to reconvert system in discrete state, 
        #but with dt = 1 because every sample must be rescaled from distance step to 1
        sys_disc_1 = harold.discretize(sys_cont, dt = 1, method = 'tustin')

        #extract matrices, they are used to reconstruct 
        A_disc_1 = sys_disc_1.a
        B_disc_1 = sys_disc_1.b       
        C_disc_1 = sys_disc_1.c
        D_disc_1 = sys_disc_1.d

        #extraction of eigenvalues with dt = 1
        [dmdc.mu_SLOW_1,eigenvectors] = eig(A_disc_1)

       

        #plot of A_disc_1, A_disc_step, A_cont
        x = A_disc_1.shape[0]
        y = A_disc_1.shape[1]
        x = np.linspace(0, x, x)
        y = np.linspace(0, y, y)
        #make_plot(A_disc_1, x=y, y=x, title=   'A_disc_1' + " nyq: " + str(nyq) + " step: " + str(step), figsize=(7.5, 5), xlabel = 'State', ylabel = 'Output')

        x = A_disc_step.shape[0]
        y = A_disc_step.shape[1]
        x = np.linspace(0, x, x)
        y = np.linspace(0, y, y)
        #make_plot(A_disc_step, x=y, y=x, title="A_disc_step nyq: " + str(nyq) + " step: " + str(step), figsize=(7.5, 5), xlabel = 'State', ylabel = 'Output')

        x = A_cont.shape[0]
        y = A_cont.shape[1]
        x = np.linspace(0, x, x)
        y = np.linspace(0, y, y)
        #make_plot(A_cont, x=y, y=x, title="A_cont nyq: " + str(nyq) + " step: " + str(step), figsize=(7.5, 5), xlabel = 'State', ylabel = 'Output')


        '''
        Now the algorithm have the matrices A and B to compute the reconstructed matrix for the node (DMDc object). 
        The reconstruction for one node starts from the initial conditions (delcared as an array). 
        This array will become a matrix after the reconstruction code block
        '''

       #parameters of the reconstruction
        D_dmdc = [D[:, 0]].copy()

        D_dmdc_A = D_dmdc.copy()    #these are used for the test (work in progress)
        D_dmdc_B = [] 

        #number of rows expected after the reconstruction
        expected_shape = D_dmdc[0].shape
        

        #the reconstruction continue with the calculation about data and inputs like DMDc algorithm 
        #but with the difference that the reconstrution is with all inputs and not only subsampled inputs
        #data[i] represents the i-th instant of the state data, u represents the i-th instant of the input data




        '''reconstrcution with A and B with dt = step'''
        if reconstruction == 1:
            for i, u in enumerate(U[:,1:].T):            

                d_succ = A_disc_step.dot(D_dmdc[i]) + B_disc_step.dot(u)

                if d_succ.shape != expected_shape:
                    raise ValueError(
                        f"Invalid shape: expected {expected_shape}, got {d_succ.shape}"
                    ) 
                D_dmdc.append(d_succ)
                D_dmdc_A.append(A_disc_step.dot(D_dmdc[i]))
                D_dmdc_B.append(B_disc_step.dot(u))

            D_dmdc = np.array(D_dmdc).T
            D_dmdc_A = np.array(D_dmdc_A).T
            D_dmdc_B = np.array(D_dmdc_B).T






        '''Reconstruction with A and B with dt = 1'''
        if reconstruction == 2:
            for i, u in enumerate(U[:,1:].T):        

                d_succ = ((A_disc_1.dot(D_dmdc[i]) + B_disc_1.dot(u)))

                if d_succ.shape != expected_shape:
                    raise ValueError(
                        f"Invalid shape: expected {expected_shape}, got {d_succ.shape}"
                    ) 
                D_dmdc.append(d_succ)
                D_dmdc_A.append(A_disc_1.dot(D_dmdc[i]))
                D_dmdc_B.append(B_disc_1.dot(u))

            D_dmdc = np.array(D_dmdc).T
            D_dmdc_A = np.array(D_dmdc_A).T
            D_dmdc_B = np.array(D_dmdc_B).T





        #save discrete system with dt = 1
        dmdc.sys_disc_1 = sys_disc_1








        '''Reconstruction with forced responce (work in progress)'''
        if reconstruction == 3:

            D_dmdc = D[:,0]

            t_train = np.linspace(1, D.shape[1], D.shape[1])
            sys_disc_step_control = ct.StateSpace(A_disc_step, B_disc_step, C_disc, D_disc, dt = 1)
            time, D_dmdc = ct.forced_response(sys_disc_step_control, T=None, U=U[:,1:], X0 = D_dmdc, interpolate = False, transpose=False)

            sys_disc_1_control = ct.StateSpace(A_disc_1, B_disc_1, C_disc_1, D_disc_1, dt = 1)
            #time, D_dmdc = ct.forced_response(sys_disc_1_control, T=None, U=U, X0 = D_dmdc0, interpolate = False)

            t_train = np.linspace(1, D.shape[1], D.shape[1])
            sys_cont_control = ct.StateSpace(A_cont, B_cont, C_cont, D_cont)
            #time, D_dmdc = ct.forced_response(sys_cont_control, T=t_train, U=U, X0 = D_dmdc0, interpolate = True)


            '''
            t_train = np.linspace(0,D.shape[1],D.shape[1])
            plt.plot(t_train, D_dmdc[0,:], 'k', label='Forced response')
            t_train = np.linspace(0,D.shape[1],D.shape[1])
            plt.plot(t_train, D[0,:], 'b', label='Experimental data')
            plt.legend()
            plt.show()
            '''
            
            '''
            for i, u in enumerate(U[:,1:].T):      

                u = u[: , np.newaxis]
                #T=np.linspace(1,1 ,1)

                time, d_succ = ct.forced_response(sys_disc_step_control, T=None, U=u, X0 = D_dmdc[i], interpolate = True)
                
                #time, D_dmdc = signal.dlsim(sys_disc_step_signal, u)


                if d_succ.shape != expected_shape:
                    raise ValueError(
                        f"Invalid shape: expected {expected_shape}, got {d_succ.shape}"
                    ) 
                D_dmdc.append(d_succ)
                D_dmdc_A.append(A_disc_1.dot(D_dmdc[i]))
                D_dmdc_B.append(B_disc_1.dot(u))

            D_dmdc = np.array(D_dmdc).T
            D_dmdc_A = np.array(D_dmdc_A).T
            D_dmdc_B = np.array(D_dmdc_B).T
            '''
            

            # vars for the objective function for D (before subsampling)
            Vand = np.vander(power(mu_SLOW_step, 1/step), bin_size, True)   ## vander() restituisce una matrice di Vandermonde, come paramentri
                                                              ## vanno passati: un array 1-D (in questo caso mu elevato a potenza
                                                              ## 1/1600), il numero di colonne dell'uscita e un valore booleano
                                                              ## che indica l'incremento (se True allora le colonne saranno
                                                              ## x^0, x^1, x^2... se False saranno x^(N-1), x^(N-2),...)
                        
            P = multiply(dot(Phi_SLOW_step.conj().T, Phi_SLOW_step), np.conj(dot(Vand, Vand.conj().T)))  ## multiply() serve per moltiplicare due array.
                                                                                     ## in questo caso tra il [prodotto scalare della
                                                                                 ## la congiunta di phi trasposta e phi] e 
                                                                                 ## [la congiunta del prodotto scalare di Vand
                                                                                 ## e la congiunta di Vand trasposta]
            q = np.conj(diag(dot(dot(Vand, D.conj().T), Phi_SLOW_step)))    ##

            # find optimal b solution
            b_opt = solve(P, q).squeeze()                         ## b = P^-1 * q
                                                              ## solve() trova le radici di P risolvendo per q

            # time evolution
            Psi = (Vand.T * b_opt).T                              ## Psi matrice (2,1600)
        


        
      
        


    
        
        

        
        
        









        #save the continuous system to see zeros poles for every level
        dmdc.sys_cont = sys_cont
        
        

    else:
        #reassignments of A,B are made in case no SLOW features are found   
        #fast modes
        D_dmdc_A = np.zeros([D.shape[0], D.shape[1]], dtype='complex')
        D_dmdc_B = np.zeros([D.shape[0], D.shape[1]], dtype='complex')

        D_dmdc = np.zeros([D.shape[0], D.shape[1]], dtype='complex')

        b_opt = np.array([], dtype='complex')
        Psi = np.zeros([0, bin_size], dtype='complex')

   
    dmdc.D_dmdc_A_disc_1 = D_dmdc_A
    dmdc.D_dmdc_B_disc_1 = D_dmdc_B

    
    if reconstruction == 3:
        # dmd reconstruction
        free_evolution = dot(Phi_SLOW_step, Psi)
        D_dmdc =  free_evolution 
    



    #x = D_dmdc_disc_1.shape[0]
    #y = D_dmdc_disc_1.shape[1]
    #x = np.linspace(0, x, x)
    #y = np.linspace(0, y, y)
    #make_plot(D_dmdc.T, x=x, y=y, title='levels 0-' + str(level) + " nyq: " + str(nyq) + " step: " + str(step), figsize=(7.5, 5))
    #plt.figure()
    #plt.plot(y, D_dmdc_A[0,:], 'b', label='D_dmdc_A')
    #plt.plot(y, D_dmdc_B[0,:], 'g', label='D_dmdc_B')
    #plt.legend()
    #plt.show()

    #x = np.linspace(0, D.shape[1], forced_discrete_response.shape[1])
    #plt.plot(x, forced_discrete_response[0,:], 'orange', label='Forced discrete response')
    #plt.plot(t_train, forced_discrete_step_response[0,:], 'orange', label='Forced discrete response')
    #plt.plot(time_1, forced_discrete_1_response[0,:], 'red', label='Forced discrete 1 response')
    #plt.plot(time_cont, forced_continue_response[0,:], 'k', label='Forced continue response')
    #plt.plot(time_step, forced_discrete_step_response[0,:], 'orange', label='Forced discrete step response')    
    #plt.plot(t_train, D[0,:], 'b', label='Experimental data')
    #plt.plot(t_train, D_dmdc[0,:], 'green', label='Free + Forced evolution')
    #plt.plot(t_train, free_evolution[0,:], 'red', label='Free evolution')
    #plt.legend()
    #plt.show()
    


    # remove influence of slow modes                                              
    D = D - D_dmdc

    #save matrix for the reconstruction
    dmdc.dato = D_dmdc

    #save eigenvalues and modes with dt = step
    dmdc.mu_SLOW_step = mu_SLOW_step
    dmdc.phi_SLOW_step = Phi_SLOW_step

    



    '''
    dmdc.level = level            # level of recursion
    dmdc.bin_num = bin_num        # time bin number
    dmdc.bin_size = bin_size      # time bin size
    dmdc.start = offset           # starting index
    dmdc.stop = offset + bin_size # stopping index
    dmdc.step = step              # step size
    dmdc.rho = rho                # frequency cutoff
    dmdc.n = n                    # number of extracted modes
    dmdc.mu = mu_SLOW_step                  # extracted eigenvalues
    dmdc.Phi = Phi_SLOW_step                # extracted DMD modes
    #dmdc.Psi = Psi                # extracted time evolution
    '''

    if reconstruction == 3:
        dmdc.b_opt = b_opt            # extracted optimal b vector



    #the code will iterate splitting the data and input, stopping if level = max_levels or return when bin_size < nyq
    if level < max_levels:
        split = ceil(bin_size / 2) # where to split           ## ceil(x) approximate by excess
        nodes += mrdmdc(
            D[:,:split],
            U[:,:split],                         
            level=level+1,
            bin_num=2*bin_num,
            offset=offset,
            max_levels=max_levels,
            max_cycles=max_cycles,
            )
        nodes += mrdmdc(
            D[:,split:],
            U[:,split:],                       
            level=level+1,
            bin_num=2*bin_num+1,
            offset=offset+split,
            max_levels=max_levels,
            max_cycles=max_cycles,
            )
        #With every iteration of the algorithm there is an addition of a node (that is a DMDc object) in an array
        #At the end the algorithm return the array with nodes processed in all iterations
    return nodes


nodes = mrdmdc(D_train, U_train)







def stitch(nodes, level):

    # get length of time dimension
    start = min([nd.start for nd in nodes])
    stop = max([nd.stop for nd in nodes])
    t = stop - start

    # extract relevant nodes
    nodes = [n for n in nodes if n.level == level]
    nodes = sorted(nodes, key=lambda n: n.bin_num)

    # stack DMD modes
    Phi = np.hstack([n.Phi for n in nodes])

    # allocate zero matrix for time evolution
    nmodes = sum([n.n for n in nodes])
    Psi = np.zeros([nmodes, t], dtype='complex')

    # copy over time evolution for each time bin
    i = 0
    for n in nodes:
        _nmodes = n.Psi.shape[0]
        Psi[i:i+_nmodes,n.start:n.stop] = n.Psi
        i += _nmodes

    return Phi,Psi










    


'''
def dimensionamento(dataset, column):

    #this function allow to plot the graphic of the system during his reconstruction with the dimension of the original system
    #an example a matrix 40x4 become 40x7160 

    # Calcola il fattore di ripetizione per ogni colonna
    fattore_ripetizione = math.ceil(column / dataset.shape[1])    #approssimo per eccesso 
    # Espandi le colonne della matrice
    matrice_finale = np.repeat(dataset, fattore_ripetizione, axis=1)
    # Riduci le colonne al numero desiderato
    matrice_finale = matrice_finale[:, :column]         #qui mi faccio il troncamento
    
    return matrice_finale
'''




def iteration_level(nodes):
    #this function allow to know the level of iterations that mrdmdc done
    level = 0
    for n in nodes:
        if n.level > level:
            level = n.level
    return int(level) 





# start the reconstruction of data train
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

D_train_subtracted = D_train.copy()

#declare variable that filled up in the reconstruction
D_mrdmdc = np.zeros([D_train.shape[0], D_train.shape[1]], dtype = complex)    
D_mrdmdc_A = np.zeros([D_train.shape[0], D_train.shape[1]], dtype = complex)
D_mrdmdc_B = np.zeros([D_train.shape[0], D_train.shape[1]], dtype = complex)

for level in range(0 , iteration_level(nodes) + 1):

    # extract and sort relevant nodes of single level 
    nodes_level = [n for n in nodes if n.level == level]
    nodes_level = sorted(nodes_level, key=lambda n: n.bin_num)

    #save step and nyq values, they can be seen in the plots
    nyq = nodes_level[0].nyq
    step = nodes_level[0].step

    #horizontal stack of reconstructed matrices of nodes with same level (stack reconstructed snapshots in order)
    D_mrdmdc_level_reconstruction = np.hstack([n.dato for n in nodes_level])
    
    #sum every reconstructed level 
    D_mrdmdc += D_mrdmdc_level_reconstruction


    #plots of the reconstruction matrix and of the first column (during the reconstruction)
    x = D_mrdmdc.shape[0]
    y = D_mrdmdc.shape[1]
    x = np.linspace(0, x, x)
    y = np.linspace(0, y, y)
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    make_plot(D_mrdmdc.T, x=x, y=y, title= 'Reconstructed system, levels 0-' + str(level) + ", nyq: " + str(nyq) + ", step: " + str(step), xlabel = matrix_xlabel, ylabel = matrix_ylabel)
    plt.subplot(122)
    comparison(D_train_subtracted, D_mrdmdc_level_reconstruction, level, title = 'levels 0-' + str(level) + ", step: " + str(step))
    D_train_subtracted -= D_mrdmdc_level_reconstruction.real 
    plt.show()




#this part helps to see all reconstructed matrices of levels in one plot 
D_mrdmdc = np.zeros([D_train.shape[0], D_train.shape[1]], dtype = complex)    
plt.figure()

for level in range(0 , iteration_level(nodes) + 1):

    # extract relevant nodes
    nodes_level = [n for n in nodes if n.level == level]
    nodes_level = sorted(nodes_level, key=lambda n: n.bin_num)

    D_mrdmdc_level_reconstruction = np.hstack([n.dato for n in nodes_level])

    #writing_csv(path_for_save_reconstructed_train + '_level_' + str(level) + '.csv', D_mrdmdc_level_reconstruction)

    plt.plot(t_train, D_mrdmdc_level_reconstruction[column_to_show,:], color=colors[level], label='D_mrdmdc level: ' + str(level))

    D_mrdmdc += D_mrdmdc_level_reconstruction

    nyq = nodes_level[0].nyq
    step = nodes_level[0].step

    filtration = []
    for node in nodes_level:
        filtration.extend([node.percentage_of_filtration])
    percentage_of_filtration = sum(filtration) // len(filtration)
    print('filtration: ' + str(percentage_of_filtration) + '%' + ' level: ' + str(level))

#plt.plot(t_train, D_mrdmdc[column_to_show,:], 'k', label='Reconstructed system')

plt.plot(t_train, D_train[column_to_show,:], 'g', label='Experiemental system')
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.title("Signal of each level state: " + array_title)
plt.legend()
plt.show()



#this part helps to see A*X and B*U of each levels
'''
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1)) 

for level in range (0, iteration_level(nodes) + 1):
    nodes_level = [n for n in nodes if n.level == level]
    nodes_level = sorted(nodes_level, key=lambda n: n.bin_num)

    D_mrdmdc_level_A = np.hstack([n.D_dmdc_A_disc_1 for n in nodes_level])
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.plot(t_train, D_mrdmdc_level_A[0,:], color=colors[level], label='D_mrdmdc_A level: ' + str(level))
    plt.legend()

    D_mrdmdc_level_B = np.hstack([n.D_dmdc_B_disc_1 for n in nodes_level])
    plt.subplot(122)
    plt.plot(t_train, D_mrdmdc_level_B[0,:], color=colors[level], label='D_mrdmdc_B level: ' + str(level))
    plt.legend()
    plt.show()
'''


writing_csv(path_for_save_reconstructed_train + '_reconstructed_train.csv', D_mrdmdc.real)
writing_csv(path_for_save_reconstructed_train + '_experimental_train.csv' , D_train.real)






#Comparison between original data and reconstructed data
plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.title('Experimental system')
plt.pcolor(D_train.real.T, vmin = vmin, vmax = vmax)
plt.colorbar()
plt.xlabel(matrix_xlabel)
plt.ylabel(matrix_ylabel)

plt.subplot(122)
plt.title('Reconstructed system')
plt.pcolor(D_mrdmdc.real.T, vmin = vmin, vmax = vmax)
plt.colorbar()
plt.xlabel(matrix_xlabel)
plt.ylabel(matrix_ylabel)

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
print((MSE(D_mrdmdc.T,D_train.T)))
print("{:.2e}".format(MSE(D_mrdmdc.T , D_train.T)))

print ("MAPE: ")
print (MAPE(D_mrdmdc.T , D_train.T))
print("{:.2e}".format(MAPE(D_mrdmdc.T , D_train.T)))

print ("MAE: ")
print(MAE(D_mrdmdc.T , D_train.T))
print("{:.2e}".format(MAE(D_mrdmdc.T , D_train.T)))

print ("RMSE: ")
print(RMSE(D_mrdmdc.T , D_train.T))
print("{:.2e}".format(RMSE(D_mrdmdc.T , D_train.T)))

print ("R2: ")
print(R2(D_mrdmdc.T , D_train.T))
print("{:.2e}".format(R2(D_mrdmdc.T , D_train.T)))


#consider only one state to compare both
D_state_train = D_train[column_to_show,:]
D_mrDMDc_state = D_mrdmdc[column_to_show,:]

plt.figure()
plt.title(array_title)
plt.plot(t_train, D_state_train.real, 'b', label='Experimental data')
plt.plot(t_train, D_mrDMDc_state.real, 'g', label='mrDMDc reconstructed data')
plt.xlabel(array_xlabel)
plt.ylabel(array_ylabel)
plt.legend()
plt.show()

plt.figure()
plt.title(array_title)
error=np.array(D_state_train) - np.array(D_mrDMDc_state.real)
plt.plot(t_train, error, 'b', label='Error')
plt.xlabel(array_xlabel)
plt.ylabel('Error')
plt.legend()
plt.show()








# helps to set 0 value with withe colour
def max_value(matrix):
    max_value = np.max([np.max(matrix)])
    min_value = np.min([np.min(matrix)])
    if (max_value >= -(min_value)):
        scale = max_value
    else:
        scale = -min_value
    return scale



'''
#plot of B, A_tilde, A for each level
for level in range (0, iteration_level(nodes) + 1):
    nodes_level_B = []
    nodes_level_A_tilde = []
    nodes_level_A = []
    for n in nodes:
        if n.level == level:
            nodes_level_B.append(n.B.real)
            nodes_level_A_tilde.append(n._Atilde._Atilde)       # n.A_tilde is an object of class DMDcUnkonwnOperator, n.A_tilde.A_tilde is the matrix
            nodes_level_A.append(n.A)
 
    sum_level_B = sum(nodes_level_B)
    mean_level_B = sum_level_B / len(nodes_level_B)
    x = np.linspace(0, mean_level_B.shape[0], mean_level_B.shape[0])
    y = np.linspace(0, mean_level_B.shape[1], mean_level_B.shape[1])
    
    scale_B = max_value(mean_level_B)
    #make_plot(mean_level_B, x=y, y=x, title = 'Mean _B level: ' + str(level), xlabel = 'Input', ylabel = 'Output', vmin = -(scale_B), vmax = (scale_B), ticks = 40)    #self.B

    #the sum of A_tilde works only if A_tilde is not truncated (svd_rank = -1) and also if the parameter max_cycles allow to have _D with shape[1] > shape[2] 
    
    #sum_level_A_tilde = sum(nodes_level_A_tilde)
    #mean_level_A_tilde = sum_level_A_tilde / len(nodes_level_A_tilde)
    #x = np.linspace(0, mean_level_A_tilde.shape[0], mean_level_A_tilde.shape[0])
    #y = np.linspace(0, mean_level_A_tilde.shape[1], mean_level_A_tilde.shape[1])

    #scale_A_tilde = max_value(mean_level_A_tilde)
    #make_plot(mean_level_A_tilde, x=y, y=x, title = 'Mean A_tilde level: ' + str(level), xlabel = 'State', ylabel = 'Output', vmin = -(scale_A_tilde), vmax = (scale_A_tilde))   #self._Atilde
    

    
    sum_level_A = sum(nodes_level_A)
    mean_level_A = sum_level_A / len(nodes_level_A)
    x = np.linspace(0, mean_level_A.shape[0], mean_level_A.shape[0])
    y = np.linspace(0, mean_level_A.shape[1], mean_level_A.shape[1])

    scale_A = max_value(mean_level_A)
    #make_plot(mean_level_A, x=y, y=x, title = 'Mean A level: ' + str(level), xlabel = 'State', ylabel = 'Output', vmin = -(scale_A), vmax = (scale_A))   #self.A
'''
    




#plot of A_tilde and B for each node
for level in range (0, 3):
    count = 0
    for node in nodes:
        if node.level == level:
            x = np.linspace(0, node.B.shape[0], node.B.shape[0])
            y = np.linspace(0, node.B.shape[1], node.B.shape[1])

            scale_B = max_value(node.B)
            #make_plot(node.B, x=y, y=x, title='B level: ' + str(level) + ' Node: ' + str(count), xlabel = 'Input', ylabel = 'Output',  vmin = -(scale_B), vmax = (scale_B))

            x = np.linspace(0, node._Atilde._Atilde.shape[0], node._Atilde._Atilde.shape[0])
            y = np.linspace(0, node._Atilde._Atilde.shape[1], node._Atilde._Atilde.shape[1])

            scale_A_tilde = max_value(node._Atilde._Atilde)
            #make_plot(node._Atilde._Atilde, x=y, y=x, title='A_tilde level: ' + str(level) + ' Node: ' + str(count), xlabel = 'State', ylabel = 'Output', vmin = -(scale_A_tilde), vmax = (scale_A_tilde))
        count = count + 1






#plots of eigenvalues with dt = step before the SLOW filtration and after, with the percentage of filtration for each levels, one level in one plot
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))
for level in range (0, iteration_level(nodes) + 1):
    #declaration of lists that will filled up for the plots
    eigenvalues = []
    eigenvalues_SLOW = []
    filtration = []
    for node in nodes:
        if node.level == level:
            #when node.level == level save eigenvalues before the filtration and after it, this allow to plot all eigenvalues of each level in the same plot
            step = node.step
            rho = node.rho
            eigenvalues.extend((np.abs(np.log(node.eigs) / (2 * pi * step))))
            eigenvalues_SLOW.extend((np.abs(np.log(node.mu_SLOW_step) / (2 * pi * step))))
            filtration.extend([node.percentage_of_filtration])


    real_part_eigenvalues = np.real(eigenvalues)
    imag_part_eigenvalues = np.imag(eigenvalues)

    real_part_eigenvalues_SLOW = np.real(eigenvalues_SLOW)
    imag_part_eigenvalues_SLOW = np.imag(eigenvalues_SLOW)

    percentage_of_filtration = sum(filtration) / len(filtration)
    
    
    plt.figure(figsize=(8, 8))

    #also plot the circle to see wich eigenvalues was chosen like SLOW
    circle = plt.Circle([0,0], radius = rho, fill = False)
    plt.gca().add_patch(circle)

    plt.scatter(real_part_eigenvalues, imag_part_eigenvalues, marker='x', color=colors[level], label='Eigen')
    plt.scatter(real_part_eigenvalues_SLOW, imag_part_eigenvalues_SLOW, color = 'black', marker='x', label='Eigen SLOW')

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Eigenvalues map level: ' + str(level) + ' filtration: ' + str(percentage_of_filtration) + '%' + ' step:' + str(step))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.legend()

    plt.grid(True)
    plt.show()







#plot of eigenvalues and eigenvalues SLOW with dt = step, all levels in one single plot
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

plt.figure(figsize=(8, 8))
plt.scatter([],[], color = 'black', label = "eigenvalues SLOW")

for level in range(iteration_level(nodes) + 1):
    plt.scatter([],[], color = colors[level], label = "eigenvalues level: " + str(level))
    for node in nodes:
        if node.level == level:
            plt.scatter(node.eigs.real, node.eigs.imag, marker='x', color=colors[level])
            plt.scatter(node.mu_SLOW_step.real, node.mu_SLOW_step.imag, color = 'black', marker='x')


plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.title('Eigenvalues map of levels')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.legend()

plt.grid(True)
plt.show()


















#plot of only eigenvalues SLOW with dt = step, each level in a single plot
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))
for level in range (0, iteration_level(nodes) + 1):
    eigenvalues = []
    filtration = []
    for node in nodes:
        if node.level == level:
            eigenvalues.extend(node.mu_SLOW_step)
            filtration.extend([node.percentage_of_filtration])
            step = node.step

    real_part = np.real(eigenvalues)
    imag_part = np.imag(eigenvalues)

    percentage_of_filtration = sum(filtration) / len(filtration)

    plt.figure(figsize=(8, 8))
    plt.scatter(real_part, imag_part, marker='x', color=colors[level], label='Eigen')

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Eigenvalues SLOW map dt = step level: ' + str(level) + ' filtration: ' + str(percentage_of_filtration) + '%' + ' step' + str(step))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.legend()

    plt.grid(True)
    plt.show()







#plot of all eigenvalues SLOW with dt = step of all levels in a single plot 
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

plt.figure(figsize=(8, 8))

for level in range(iteration_level(nodes) + 1):
    plt.scatter([],[], color = colors[level], label = "eigenvalues level: " + str(level))
    for node in nodes:
        if node.level == level:
            #plt.scatter(node.eigs.real / (2* pi * node.step), node.eigs.imag / (2 * pi * node.step), marker='x', color=colors[level])
            plt.scatter(node.mu_SLOW_step.real, node.mu_SLOW_step.imag, marker='x', color=colors[level])


plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.title('Eigenvalues SLOW map dt = step')
plt.xlabel('Asse Reale')
plt.ylabel('Asse Immaginario')
plt.legend()

plt.grid(True)
plt.show()









#plot of eigenvalues SLOW of each level in continuous
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))
for level in range (0, iteration_level(nodes) + 1):
    eigenvalues = []
    #filtration = []
    for node in nodes:
        if node.level == level:
            step = node.step
            if node.mu_SLOW_cont is None:
                break
            eigenvalues.extend(node.mu_SLOW_cont)
            #filtration.extend([node.percentage_of_filtration])

    real_part = np.real(eigenvalues)
    imag_part = np.imag(eigenvalues)

    #if len(filtration) != 0:
    #   percentage_of_filtration = sum(filtration) / len(filtration)

    plt.figure(figsize=(8, 8))
    plt.scatter(real_part, imag_part, marker='x', color=colors[level], label='Eigen')

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Eigenvalues SLOW map in continuous level: ' + str(level) + ' step:' +str(step))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.legend()

    plt.grid(True)
    plt.show()



    
#plot of all eigenvalues SLOW in continuous in the same plot 
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

plt.figure(figsize=(8, 8))

for level in range(iteration_level(nodes) + 1):
    plt.scatter([],[], color = colors[level], label = "eigenvalues level: " + str(level))
    for node in nodes:
        if node.level == level:
            if node.mu_SLOW_cont is None:
                break
            plt.scatter(node.mu_SLOW_cont.real, node.mu_SLOW_cont.imag, marker='x', color=colors[level])

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.title('Eigenvalues SLOW map in continuous')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.legend()

plt.grid(True)
plt.show()















#plot of eigenvalues SLOW of each level with dt = 1
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

for level in range (0, iteration_level(nodes) + 1):
    eigenvalues = []
    #filtration = []
    for node in nodes:
        if node.level == level:
            step = node.step
            if node.mu_SLOW_cont is None:
                break
            eigenvalues.extend(node.mu_SLOW_1)
            #filtration.extend([node.percentage_of_filtration])

    real_part = np.real(eigenvalues)
    imag_part = np.imag(eigenvalues)

    #if len(filtration) != 0:
    #   percentage_of_filtration = sum(filtration) / len(filtration)

    plt.figure(figsize=(8, 8))
    plt.scatter(real_part, imag_part, marker='x', color=colors[level], label='Eigen')

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Eigenvalues SLOW map dt = 1 level: ' + str(level) + ' step:' +str(step))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.legend()

    plt.grid(True)
    plt.show()




#plot of all eigenvalues SLOW with dt = 1 in the same plot 
colors = plt.cm.rainbow(np.linspace(0, 1, iteration_level(nodes) + 1))

plt.figure(figsize=(8, 8))

for level in range(iteration_level(nodes) + 1):
    plt.scatter([],[], color = colors[level], label = "eigenvalues level: " + str(level))
    for node in nodes:
        if node.level == level:
            if node.mu_SLOW_cont is None:
                break
            plt.scatter(node.mu_SLOW_1.real, node.mu_SLOW_1.imag, marker='x', color=colors[level])

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.title('Eigenvalues SLOW map dt = 1')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.legend()

plt.grid(True)
plt.show()














'''

Facciamo una funzione per controllare qual è il bin_size più vicino alla U_test

def choose_bin(U_test, nodes):
    for i in range(0, len(nodes)):
        if (nodes[i].bin_size >= U_test.shape[1]):
            node = nodes[i]
    return node

A questo punto vado a campionare la U_test e la passo a reconstructed_data

dmdc_reconstruct = choose_bin(U_test, nodes)
i = dmdc_reconstruct.level
U_test = dimensionamento(U_test, dmdc_reconstruct.bin_size)
U_test_sub = U_test[:,::dmdc_reconstruct.step]

nodes_reconstruct = []
for node in nodes:
    if node.level == dmdc_reconstruct.level:
        nodes_reconstruct.append(node)

#rec = np.zeros([row_tot, column_train], dtype = complex)
insieme_reconstruct = []

for node in nodes_reconstruct:
    reconstructed_data_test = node.reconstructed_data(U_test_sub[:,1:])
    insieme_reconstruct.append(reconstructed_data_test)      #append function add the data passed to array

reconstruct = np.hstack(insieme_reconstruct)

#reconstruct = dimensionamento(reconstruct, column_train)
x = reconstruct.shape[0]
y = reconstruct.shape[1]   
x = np.linspace(0, x, x)
y = np.linspace(0, y, y)
make_plot(reconstruct.T, x=x, y=y, title='level: ' + str(i), figsize=(7.5, 5))
#confronto(D_train, reconstruct)


'''


