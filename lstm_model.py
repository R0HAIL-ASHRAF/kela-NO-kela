import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

        init_scale = np.sqrt(2.0 / (input_size + hidden_size))
        out_scale = np.sqrt(2.0 / (hidden_size + output_size))

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * init_scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * init_scale

        self.Wy = np.random.randn(output_size, hidden_size) * out_scale

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.m = {}
        self.v = {}
        self.param_names = ['Wf', 'Wi', 'Wc', 'Wo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']

        for name in self.param_names:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

        self.t = 0

        self.x_cache = {}
        self.h_cache = {}
        self.c_cache = {}
        self.f_cache = {}
        self.i_cache = {}
        self.c_tilde_cache = {}
        self.o_cache = {}

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def stable_softmax(self, z):
        shift_z = z - np.max(z)
        exps = np.exp(shift_z)
        return exps / np.sum(exps)

    def forward(self, x_sequence, training=False):
        T = x_sequence.shape[0]

        self.h_cache[-1] = np.zeros((self.hidden_size, 1))
        self.c_cache[-1] = np.zeros((self.hidden_size, 1))

        for t in range(T):
            xt = x_sequence[t].reshape(-1, 1)
            self.x_cache[t] = xt

            concat = np.vstack((xt, self.h_cache[t-1]))

            self.f_cache[t] = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            self.i_cache[t] = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            self.c_tilde_cache[t] = np.tanh(np.dot(self.Wc, concat) + self.bc)
            self.o_cache[t] = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            self.c_cache[t] = self.f_cache[t] * self.c_cache[t-1] + self.i_cache[t] * self.c_tilde_cache[t]
            self.h_cache[t] = self.o_cache[t] * np.tanh(self.c_cache[t])

        h_out = self.h_cache[T-1]
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*h_out.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            h_out = h_out * self.dropout_mask
        else:
            self.dropout_mask = np.ones_like(h_out)

        z = np.dot(self.Wy, h_out) + self.by
        return self.stable_softmax(z)

    def backward(self, y_pred, y_true_idx, lr, l2_lambda=0.0001):
        T = len(self.x_cache)

        self.dWf, self.dWi, self.dWc, self.dWo, self.dWy = [np.zeros_like(w) for w in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy]]
        self.dbf, self.dbi, self.dbc, self.dbo, self.dby = [np.zeros_like(b) for b in [self.bf, self.bi, self.bc, self.bo, self.by]]

        dz = y_pred.copy()
        dz[y_true_idx] -= 1.0

        h_out = self.h_cache[T-1] * self.dropout_mask
        self.dWy = np.dot(dz, h_out.T)
        self.dby = dz

        dh_next = np.dot(self.Wy.T, dz) * self.dropout_mask
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            concat = np.vstack((self.x_cache[t], self.h_cache[t-1]))
            dh = dh_next

            do = dh * np.tanh(self.c_cache[t])
            do_raw = do * self.o_cache[t] * (1.0 - self.o_cache[t])

            dc = dh * self.o_cache[t] * (1.0 - np.tanh(self.c_cache[t])**2) + dc_next

            dc_tilde = dc * self.i_cache[t]
            dc_tilde_raw = dc_tilde * (1.0 - self.c_tilde_cache[t]**2)

            di = dc * self.c_tilde_cache[t]
            di_raw = di * self.i_cache[t] * (1.0 - self.i_cache[t])

            df = dc * self.c_cache[t-1]
            df_raw = df * self.f_cache[t] * (1.0 - self.f_cache[t])

            self.dWf += np.dot(df_raw, concat.T)
            self.dWi += np.dot(di_raw, concat.T)
            self.dWc += np.dot(dc_tilde_raw, concat.T)
            self.dWo += np.dot(do_raw, concat.T)

            self.dbf += df_raw
            self.dbi += di_raw
            self.dbc += dc_tilde_raw
            self.dbo += do_raw

            dconcat = (np.dot(self.Wf.T, df_raw) +
                       np.dot(self.Wi.T, di_raw) +
                       np.dot(self.Wc.T, dc_tilde_raw) +
                       np.dot(self.Wo.T, do_raw))

            dh_next = dconcat[self.input_size:, :]
            dc_next = self.f_cache[t] * dc

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grads = [self.dWf, self.dWi, self.dWc, self.dWo, self.dWy, self.dbf, self.dbi, self.dbc, self.dbo, self.dby]

        for name, grad in zip(self.param_names, grads):
            np.clip(grad, -1.0, 1.0, out=grad)

            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad ** 2)

            m_hat = self.m[name] / (1.0 - beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - beta2 ** self.t)

            current_weight = getattr(self, name)
            l2_term = l2_lambda * current_weight if name.startswith('W') else 0.0
            updated_weight = current_weight - lr * (m_hat / (np.sqrt(v_hat) + eps) + l2_term)
            setattr(self, name, updated_weight)

    def predict(self, x_sequence):
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
