import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 1. Xavier/Glorot Initialization: Balances activation variance across steps
        init_scale = np.sqrt(2.0 / (input_size + hidden_size))
        out_scale = np.sqrt(2.0 / (hidden_size + output_size))

        # Gate Weights (Concatenated dimensions: input + hidden)
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        
        # Output Projection Weights
        self.Wy = np.random.randn(output_size, hidden_size) * out_scale

        # Biases initialized to zero vectors
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # 2. Adam Optimizer Moment Storage
        self.m = {}
        self.v = {}
        self.param_names = ['Wf', 'Wi', 'Wc', 'Wo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']
        
        for name in self.param_names:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))
        
        self.t = 0  # Adam timestep step tracker

        # Hidden state sequence tracking caches for BPTT
        self.x_cache = {}
        self.h_cache = {}
        self.c_cache = {}
        self.f_cache = {}
        self.i_cache = {}
        self.c_tilde_cache = {}
        self.o_cache = {}

    def sigmoid(self, x):
        # Numerically stable activation boundary capping
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def stable_softmax(self, z):
        # Maximum element offset subtraction prevents infinity (NaN) crashes
        shift_z = z - np.max(z)
        exps = np.exp(shift_z)
        return exps / np.sum(exps)

    def forward(self, x_sequence):
        T = x_sequence.shape[0]
        
        # Establish temporal boundaries base values
        self.h_cache[-1] = np.zeros((self.hidden_size, 1))
        self.c_cache[-1] = np.zeros((self.hidden_size, 1))

        for t in range(T):
            xt = x_sequence[t].reshape(-1, 1)
            self.x_cache[t] = xt

            # Structural concatenation setup
            concat = np.vstack((xt, self.h_cache[t-1]))

            # Evaluate cell gating operations
            self.f_cache[t] = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            self.i_cache[t] = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            self.c_tilde_cache[t] = np.tanh(np.dot(self.Wc, concat) + self.bc)
            self.o_cache[t] = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            # Update memory streams
            self.c_cache[t] = self.f_cache[t] * self.c_cache[t-1] + self.i_cache[t] * self.c_tilde_cache[t]
            self.h_cache[t] = self.o_cache[t] * np.tanh(self.c_cache[t])

        # Push final timestep activation out to category distribution map
        z = np.dot(self.Wy, self.h_cache[T-1]) + self.by
        return self.stable_softmax(z)

    def backward(self, y_pred, y_true_idx, lr):
        T = len(self.x_cache)

        # Initialize parameter derivative sheets
        self.dWf, self.dWi, self.dWc, self.dWo, self.dWy = [np.zeros_like(w) for w in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy]]
        self.dbf, self.dbi, self.dbc, self.dbo, self.dby = [np.zeros_like(b) for b in [self.bf, self.bi, self.bc, self.bo, self.by]]

        # Categorical cross-entropy error layer gradient
        dz = y_pred.copy()
        dz[y_true_idx] -= 1.0

        # Output projection parameter steps
        self.dWy = np.dot(dz, self.h_cache[T-1].T)
        self.dby = dz

        # Structural temporal gradient initialization
        dh_next = np.dot(self.Wy.T, dz)
        dc_next = np.zeros((self.hidden_size, 1))

        # Backpropagation Through Time Loop (BPTT)
        for t in reversed(range(T)):
            concat = np.vstack((self.x_cache[t], self.h_cache[t-1]))
            dh = dh_next
            
            # Output gating gradient distribution path
            do = dh * np.tanh(self.c_cache[t])
            do_raw = do * self.o_cache[t] * (1.0 - self.o_cache[t])

            # Memory block internal error tracking maps
            dc = dh * self.o_cache[t] * (1.0 - np.tanh(self.c_cache[t])**2) + dc_next
            
            dc_tilde = dc * self.i_cache[t]
            dc_tilde_raw = dc_tilde * (1.0 - self.c_tilde_cache[t]**2)

            di = dc * self.c_tilde_cache[t]
            di_raw = di * self.i_cache[t] * (1.0 - self.i_cache[t])

            df = dc * self.c_cache[t-1]
            df_raw = df * self.f_cache[t] * (1.0 - self.f_cache[t])

            # Sum total changes across sequence steps
            self.dWf += np.dot(df_raw, concat.T)
            self.dWi += np.dot(di_raw, concat.T)
            self.dWc += np.dot(dc_tilde_raw, concat.T)
            self.dWo += np.dot(do_raw, concat.T)

            self.dbf += df_raw
            self.dbi += di_raw
            self.dbc += dc_tilde_raw
            self.dbo += do_raw

            # Project step values upstream to clear time boundary t-1
            dconcat = (np.dot(self.Wf.T, df_raw) + 
                       np.dot(self.Wi.T, di_raw) + 
                       np.dot(self.Wc.T, dc_tilde_raw) + 
                       np.dot(self.Wo.T, do_raw))
            
            dh_next = dconcat[self.input_size:, :]
            dc_next = self.f_cache[t] * dc

        # 3. ⚡ Integrated Adam Parameter Optimization Logic
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grads = [self.dWf, self.dWi, self.dWc, self.dWo, self.dWy, self.dbf, self.dbi, self.dbc, self.dbo, self.dby]

        for name, grad in zip(self.param_names, grads):
            # Clip gradient values to counter potential remaining outliers
            np.clip(grad, -0.2, 0.2, out=grad)
            
            # Update moment tracking components
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad ** 2)
            
            # Formulate bias corrections
            m_hat = self.m[name] / (1.0 - beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - beta2 ** self.t)
            
            # Apply final adaptive scaled parameter modifications
            current_weight = getattr(self, name)
            updated_weight = current_weight - lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, updated_weight)

    def predict(self, x_sequence):
        """
        Isolated inference computation path.
        Prevents BPTT cache overrides when testing validation points.
        """
        T = x_sequence.shape[0]
        h_local = np.zeros((self.hidden_size, 1))
        c_local = np.zeros((self.hidden_size, 1))

        for t in range(T):
            xt = x_sequence[t].reshape(-1, 1)
            concat = np.vstack((xt, h_local))

            f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            c_local = f * c_local + i * c_tilde
            h_local = o * np.tanh(c_local)

        z = np.dot(self.Wy, h_local) + self.by
        return self.stable_softmax(z)