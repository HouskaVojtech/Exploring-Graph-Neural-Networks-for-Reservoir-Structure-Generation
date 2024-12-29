import numpy as np

class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=1.25, sparsity=0.1, alpha = 1, noise=0.001, random_state=None):
        """
        Initialize the Echo State Network.
        
        Parameters:
        - n_input: Number of input units.
        - n_reservoir: Number of reservoir neurons.
        - n_output: Number of output units.
        - spectral_radius: Scaling factor for the reservoir weights.
        - sparsity: Fraction of weights set to zero in the reservoir.
        - noise: Amount of noise to add to the reservoir dynamics.
        - random_state: Random seed for reproducibility.
        """
        # just for the random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        # this number is used for making the sparsity
        self.sparsity = sparsity
        self.alpha = alpha
        
        # pbbly for the noise ammount
        self.noise = noise

        # Input weight matrix
        self.W_in = np.random.uniform(-1, 1, (n_reservoir, n_input + 1))  # +1 for bias term

        # Reservoir weight matrix (random sparse matrix)
        W_res = np.random.rand(n_reservoir, n_reservoir) - 0.5
        # ah ok so you 0 out every place that has more thant sparsity based on a normaly distributed matrix
        self.sparsity_mask=np.random.rand(*W_res.shape) > sparsity
        W_res[self.sparsity_mask] = 0
        # Rescale to ensure the spectral radius
        rho_W = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (spectral_radius / rho_W)
        
        

        # Output weight matrix (initialized later)
        self.W_out = None
        
    def get_reservoir(self):
        return self.W_res
    
    def get_w_in(self):
        return self.W_in
    
    def set_reservoir(self, res):
        self.W_res = res
    
    def set_w_in(self,w_in):
        self.W_in = w_in

    def _update_reservoir(self, x, r):
        """
        Update the reservoir state.
        
        Parameters:
        - x: Current input.
        - r: Previous reservoir state.
        
        Returns:
        - Updated reservoir state.
        """
        input_with_bias = np.hstack([1.0, x])  # Add bias term to input
        # input_wiht_bias.shape == 1,self.n_input+1 
        # W_in.shape == self.n_reservoir, self.n_input+1
        # np.dot(self.W_in, input_with_bias).shape == self.n_reservoir
        # r_new.shape == self.n_reservoir
        r_new = np.tanh(np.dot(self.W_in, input_with_bias) + np.dot(self.W_res, r))# + self.noise * np.random.randn(self.n_reservoir))

        r_new = ( 1 - self.alpha ) * r + self.alpha * r_new
        return r_new

    def _update_reservoir2(self, x, r):
        """
        Update the reservoir state.
        
        Parameters:
        - x: Current input.
        - r: Previous reservoir state.
        
        Returns:
        - Updated reservoir state.
        """
        input_with_bias = np.hstack([1.0, x])  # Add bias term to input
        # input_wiht_bias.shape == 1,self.n_input+1 
        # W_in.shape == self.n_reservoir, self.n_input+1
        # np.dot(self.W_in, input_with_bias).shape == self.n_reservoir
        # r_new.shape == self.n_reservoir
        r_new = np.tanh(np.dot(self.W_in, input_with_bias) + np.dot(self.W_res, r))# + self.noise * np.random.randn(self.n_reservoir))

        #r_new = ( 1 - self.alpha ) * r + self.alpha * r_new
        return r_new

    def fit(self, X, y, ridge_reg=1e-6):
        """
        Train the output weights using ridge regression.
        
        Parameters:
        - X: Input data (n_samples, n_timesteps, n_features).
        - y: Target data (n_samples, n_timesteps, n_output).
        - ridge_reg: Regularization strength for ridge regression.
        """
        n_samples, n_timesteps, _ = X.shape
        
        # Initialize reservoir states
        reservoir_states = np.zeros((n_samples * n_timesteps, self.n_reservoir))

        # Collect reservoir states
        for i in range(n_samples):
            r = np.zeros(self.n_reservoir)  # Initial reservoir state
            for t in range(n_timesteps):
                r = self._update_reservoir(X[i, t], r)
                reservoir_states[i * n_timesteps + t] = r

        # Add bias term to reservoir states
        reservoir_states_with_bias = np.hstack([np.ones((reservoir_states.shape[0], 1)), reservoir_states])

        # Flatten target data
        target = y.reshape(-1, self.n_output)

        # Ridge regression to compute output weights
        self.W_out = np.dot(
            np.linalg.pinv(np.dot(reservoir_states_with_bias.T, reservoir_states_with_bias) + ridge_reg * np.eye(reservoir_states_with_bias.shape[1])),
            np.dot(reservoir_states_with_bias.T, target)
        )

    def predict(self, X):
        """
        Predict output for the given input data.
        
        Parameters:
        - X: Input data (n_samples, n_timesteps, n_features).
        
        Returns:
        - Predicted output (n_samples, n_timesteps, n_output).
        """
        n_samples, n_timesteps, _ = X.shape
        predictions = np.zeros((n_samples, n_timesteps, self.n_output))

        for i in range(n_samples):
            r = np.zeros(self.n_reservoir)  # Initial reservoir state
            for t in range(n_timesteps):
                r = self._update_reservoir(X[i, t], r)
                input_with_bias = np.hstack([1.0, r])  # Add bias term
                predictions[i, t] = np.dot(input_with_bias, self.W_out)

        return predictions

    def get_activations(self, X):
        n_timesteps, _ = X.shape
        
        activations = np.zeros((n_timesteps, self.n_reservoir+self.n_input+1))

        r = np.zeros(self.n_reservoir)  # Initial reservoir state
        for t in range(n_timesteps):

            r = self._update_reservoir2(X[t], r)
            inputs = np.hstack([1,X[t]]) 
            # I wan't to store these values
            activations[t] = np.concatenate([inputs, r])

        return activations


def get_signal(signal, offset, shift, window_size, max_val):
    # offset: starting position
    # shift: number of steps to shift the signal
    if window_size+shift < max_val:
        return signal[offset+shift: offset+window_size+shift].reshape(1,-1,1)
    
    return None


# In[6]:


def train_on_signal(esn, signal,offset,shift,window_size, signal_len,display=False):
    X_train = get_signal(signal, offset, shift, window_size, signal_len)
    Y_train = get_signal(signal, offset, 0, window_size, signal_len)

    if display:
        plt.plot(X_train.flatten()[:1000],label='x')
        plt.plot(Y_train.flatten()[:1000],label='y')
        plt.legend()
        plt.show()
    
    esn.fit(X_train, Y_train)
    
def test_on_signal(esn, signal, offset, shift, window_size, signal_len, display = False):
    X_test = get_signal(signal, test_offset, shift, window_size, signal_len)
    Y_test = get_signal(signal, test_offset, 0, window_size, signal_len)
    
    Y_pred = esn.predict(X_test)
    
    if display:
        plt.plot(X_test.flatten()[:500],label='x')
        plt.plot(Y_test.flatten()[:500],label='y')
        plt.plot(Y_pred.flatten()[:500],label='y_pred')
        plt.ylim(-2,2)
        plt.legend()
        plt.show()
    
    return root_mean_squared_error(Y_pred.flatten(), Y_test.flatten()), Y_pred.flatten()


# In[7]:


def get_gt(signal,offset,shift,window_size):
    X_train = get_signal(signal, offset, shift, window_size, len(signal))
    Y_train = get_signal(signal, offset, 0, window_size, len(signal))
    return X_train, Y_train

def mean_rmse(pred, ground_truth):
    total = 0
    for j,k in zip(pred,ground_truth):
        total += root_mean_squared_error(j.flatten(),k.flatten())
    return total/len(pred)


# In[11]:


def create_ecg_dataset(ecg_dataset,shift,window_size,offset=0):
    X_data = []
    Y_data = []
    for ecg in ecg_dataset:
        x_data, y_data = get_gt(ecg,offset,shift,window_size)
        X_data.append(x_data)
        Y_data.append(y_data)
        
    X_data = np.vstack(X_data).reshape(len(ecg_dataset),-1,1)
    Y_data = np.vstack(Y_data).reshape(len(ecg_dataset),-1,1)
    
    train_index = np.load('train_index.npy')
    val_index = np.load('val_index.npy')
    test_index = np.load('test_index.npy')
    
    X_train = X_data[train_index]
    Y_train = Y_data[train_index]
    
    X_val = X_data[val_index]
    Y_val = Y_data[val_index]
    
    X_test = X_data[test_index]
    Y_test = Y_data[test_index]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# In[12]:

def get_feature_vectors(X, esn, n_reservoir):
    result = esn.get_activations(X)
    # 125 * 16 = 2000 which is window size
    arr = result.reshape(125,16, n_reservoir + 2)
    return arr.mean(axis=1).T

def create_reservoirs(n_res_samples, n_reservoir, alpha, sparsity_a, sparsity_b, sr_a, sr_b ):
    results = {'mean_rmse':[],'reservoir':[],'w_in':[],'sparsity_mask':[],'sparsity':[], 'feature_vectors':[]}

    for i in range(n_res_samples):
        spectral_radius = np.random.uniform(sr_a,sr_b)
        sparsity = np.random.uniform(sparsity_a,sparsity_b)
        esn = EchoStateNetwork(n_input=1, n_reservoir=n_reservoir, n_output=1, spectral_radius=spectral_radius, alpha = alpha, sparsity=sparsity)
        esn.fit(X_train,Y_train)
        res = esn.predict(X_val)
        score = mean_rmse(res,Y_val)
        results['mean_rmse'].append(score)
        results['reservoir'].append(esn.get_reservoir())
        results['w_in'].append(esn.get_w_in())
        results['sparsity_mask'].append(esn.sparsity_mask)
        results['sparsity'].append(sparsity)
        results['feature_vectors'].append(get_feature_vectors(X_val[1], esn, n_reservoir))
        
    return results


