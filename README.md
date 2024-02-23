Utilizzo DMDc_k_step_ahead.py

I dataset sperimentali si trovano all'interno della cartella "experimental data" divisi in train e test. Bisogna copiare il percorso nelle variabili dalla riga 64 alla 73 comprensive di estensioni.
Inserire anche il path di salvataggio dei file, comprensivo dei nomi che vogliamo dare al file e della loro estensione.


Installazione delle seguenti librerie:
  matplotlib
  numpy
  scipy
  math
  pydmd
  sklearn-metrics
  csv

Installazione tramite comando: "pip install nome_libreria"
Attenzione sklearn-metrics forse va installato con "pip install scikit-metrics"

Dopo aver installato le librerie è possibile accedere al file dmdc cliccando con il tasto destro nel metodo fit o reconstructed_data e cliccando "go to definition". 

A questo punto bisgona sostituire l'implementazione di reconstructed data con quest'altra implementazione:


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
