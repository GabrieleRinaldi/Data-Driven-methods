def reconstructed_data(self, open_loop, X, control_input=None): """ Return the reconstructed data, computed using the control_input argument. If the control_input is not passed, the original input (in the fit method) is used. The input dimension has to be consistent with the dynamics.

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

    return data