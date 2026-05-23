import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        scale1 = np.sqrt(2.0 / (input_size + hidden_size))
        scale2 = np.sqrt(2.0 / (hidden_size + hidden_size))
        out_scale = np.sqrt(2.0 / (hidden_size + output_size))

        # Layer 1 weights  [hidden, input + hidden]
        self.Wf1 = np.random.randn(hidden_size, input_size + hidden_size) * scale1
        self.Wi1 = np.random.randn(hidden_size, input_size + hidden_size) * scale1
        self.Wc1 = np.random.randn(hidden_size, input_size + hidden_size) * scale1
        self.Wo1 = np.random.randn(hidden_size, input_size + hidden_size) * scale1
        self.bf1 = np.zeros((hidden_size, 1))
        self.bi1 = np.zeros((hidden_size, 1))
        self.bc1 = np.zeros((hidden_size, 1))
        self.bo1 = np.zeros((hidden_size, 1))

        # Layer 2 weights  [hidden, hidden + hidden]
        self.Wf2 = np.random.randn(hidden_size, hidden_size + hidden_size) * scale2
        self.Wi2 = np.random.randn(hidden_size, hidden_size + hidden_size) * scale2
        self.Wc2 = np.random.randn(hidden_size, hidden_size + hidden_size) * scale2
        self.Wo2 = np.random.randn(hidden_size, hidden_size + hidden_size) * scale2
        self.bf2 = np.zeros((hidden_size, 1))
        self.bi2 = np.zeros((hidden_size, 1))
        self.bc2 = np.zeros((hidden_size, 1))
        self.bo2 = np.zeros((hidden_size, 1))

        # Output layer
        self.Wy = np.random.randn(output_size, hidden_size) * out_scale
        self.by = np.zeros((output_size, 1))

        self.param_names = [
            'Wf1', 'Wi1', 'Wc1', 'Wo1', 'bf1', 'bi1', 'bc1', 'bo1',
            'Wf2', 'Wi2', 'Wc2', 'Wo2', 'bf2', 'bi2', 'bc2', 'bo2',
            'Wy', 'by'
        ]
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self.param_names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self.param_names}
        self.t = 0

        self._reset_caches()

    def _reset_caches(self):
        self.h1_cache = {}
        self.c1_cache = {}
        self.f1_cache = {}
        self.i1_cache = {}
        self.ct1_cache = {}
        self.o1_cache = {}
        self.x1_cache = {}

        self.h2_cache = {}
        self.c2_cache = {}
        self.f2_cache = {}
        self.i2_cache = {}
        self.ct2_cache = {}
        self.o2_cache = {}

        self.T = 0
        self.dropout_mask1 = None
        self.dropout_mask2 = None

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def stable_softmax(self, z):
        shift_z = z - np.max(z)
        exps = np.exp(shift_z)
        return exps / np.sum(exps)

    def forward(self, x_sequence, training=False):
        T = x_sequence.shape[0]
        self._reset_caches()
        self.T = T

        self.h1_cache[-1] = np.zeros((self.hidden_size, 1))
        self.c1_cache[-1] = np.zeros((self.hidden_size, 1))

        for t in range(T):
            xt = x_sequence[t].reshape(-1, 1)
            self.x1_cache[t] = xt
            c1 = np.vstack((xt, self.h1_cache[t - 1]))

            self.f1_cache[t] = self.sigmoid(np.dot(self.Wf1, c1) + self.bf1)
            self.i1_cache[t] = self.sigmoid(np.dot(self.Wi1, c1) + self.bi1)
            self.ct1_cache[t] = np.tanh(np.dot(self.Wc1, c1) + self.bc1)
            self.o1_cache[t] = self.sigmoid(np.dot(self.Wo1, c1) + self.bo1)

            self.c1_cache[t] = self.f1_cache[t] * self.c1_cache[t - 1] + self.i1_cache[t] * self.ct1_cache[t]
            self.h1_cache[t] = self.o1_cache[t] * np.tanh(self.c1_cache[t])

        # Variational dropout between layers (same mask across all timesteps)
        if training and self.dropout_rate > 0:
            self.dropout_mask1 = (np.random.rand(self.hidden_size, 1) > self.dropout_rate) / (1.0 - self.dropout_rate)
        else:
            self.dropout_mask1 = np.ones((self.hidden_size, 1))

        self.h2_cache[-1] = np.zeros((self.hidden_size, 1))
        self.c2_cache[-1] = np.zeros((self.hidden_size, 1))

        for t in range(T):
            h1_in = self.h1_cache[t] * self.dropout_mask1
            c2 = np.vstack((h1_in, self.h2_cache[t - 1]))

            self.f2_cache[t] = self.sigmoid(np.dot(self.Wf2, c2) + self.bf2)
            self.i2_cache[t] = self.sigmoid(np.dot(self.Wi2, c2) + self.bi2)
            self.ct2_cache[t] = np.tanh(np.dot(self.Wc2, c2) + self.bc2)
            self.o2_cache[t] = self.sigmoid(np.dot(self.Wo2, c2) + self.bo2)

            self.c2_cache[t] = self.f2_cache[t] * self.c2_cache[t - 1] + self.i2_cache[t] * self.ct2_cache[t]
            self.h2_cache[t] = self.o2_cache[t] * np.tanh(self.c2_cache[t])

        h2_out = self.h2_cache[T - 1]
        if training and self.dropout_rate > 0:
            self.dropout_mask2 = (np.random.rand(*h2_out.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            h2_out = h2_out * self.dropout_mask2
        else:
            self.dropout_mask2 = np.ones_like(h2_out)

        z = np.dot(self.Wy, h2_out) + self.by
        return self.stable_softmax(z)

    def backward(self, y_pred, y_true_idx, lr, l2_lambda=0.0001):
        T = self.T

        dWf1 = np.zeros_like(self.Wf1); dWi1 = np.zeros_like(self.Wi1)
        dWc1 = np.zeros_like(self.Wc1); dWo1 = np.zeros_like(self.Wo1)
        dbf1 = np.zeros_like(self.bf1); dbi1 = np.zeros_like(self.bi1)
        dbc1 = np.zeros_like(self.bc1); dbo1 = np.zeros_like(self.bo1)

        dWf2 = np.zeros_like(self.Wf2); dWi2 = np.zeros_like(self.Wi2)
        dWc2 = np.zeros_like(self.Wc2); dWo2 = np.zeros_like(self.Wo2)
        dbf2 = np.zeros_like(self.bf2); dbi2 = np.zeros_like(self.bi2)
        dbc2 = np.zeros_like(self.bc2); dbo2 = np.zeros_like(self.bo2)

        dz = y_pred.copy()
        dz[y_true_idx] -= 1.0

        h2_out = self.h2_cache[T - 1] * self.dropout_mask2
        dWy = np.dot(dz, h2_out.T)
        dby = dz.copy()

        dh2_next = np.dot(self.Wy.T, dz) * self.dropout_mask2
        dc2_next = np.zeros((self.hidden_size, 1))

        # Collect upstream gradients for layer 1 from all layer-2 timesteps
        dh1_from_l2 = [np.zeros((self.hidden_size, 1)) for _ in range(T)]

        # Layer 2 BPTT
        for t in reversed(range(T)):
            h1_in = self.h1_cache[t] * self.dropout_mask1
            concat2 = np.vstack((h1_in, self.h2_cache[t - 1]))
            dh2 = dh2_next

            do2 = dh2 * np.tanh(self.c2_cache[t])
            do2_r = do2 * self.o2_cache[t] * (1.0 - self.o2_cache[t])

            dc2 = dh2 * self.o2_cache[t] * (1.0 - np.tanh(self.c2_cache[t]) ** 2) + dc2_next

            dc2t = dc2 * self.i2_cache[t]
            dc2t_r = dc2t * (1.0 - self.ct2_cache[t] ** 2)

            di2 = dc2 * self.ct2_cache[t]
            di2_r = di2 * self.i2_cache[t] * (1.0 - self.i2_cache[t])

            df2 = dc2 * self.c2_cache[t - 1]
            df2_r = df2 * self.f2_cache[t] * (1.0 - self.f2_cache[t])

            dWf2 += np.dot(df2_r, concat2.T)
            dWi2 += np.dot(di2_r, concat2.T)
            dWc2 += np.dot(dc2t_r, concat2.T)
            dWo2 += np.dot(do2_r, concat2.T)
            dbf2 += df2_r; dbi2 += di2_r; dbc2 += dc2t_r; dbo2 += do2_r

            dcat2 = (np.dot(self.Wf2.T, df2_r) + np.dot(self.Wi2.T, di2_r) +
                     np.dot(self.Wc2.T, dc2t_r) + np.dot(self.Wo2.T, do2_r))

            dh1_from_l2[t] = dcat2[:self.hidden_size, :] * self.dropout_mask1
            dh2_next = dcat2[self.hidden_size:, :]
            dc2_next = self.f2_cache[t] * dc2

        # Layer 1 BPTT
        dh1_next = np.zeros((self.hidden_size, 1))
        dc1_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            concat1 = np.vstack((self.x1_cache[t], self.h1_cache[t - 1]))
            dh1 = dh1_from_l2[t] + dh1_next

            do1 = dh1 * np.tanh(self.c1_cache[t])
            do1_r = do1 * self.o1_cache[t] * (1.0 - self.o1_cache[t])

            dc1 = dh1 * self.o1_cache[t] * (1.0 - np.tanh(self.c1_cache[t]) ** 2) + dc1_next

            dc1t = dc1 * self.i1_cache[t]
            dc1t_r = dc1t * (1.0 - self.ct1_cache[t] ** 2)

            di1 = dc1 * self.ct1_cache[t]
            di1_r = di1 * self.i1_cache[t] * (1.0 - self.i1_cache[t])

            df1 = dc1 * self.c1_cache[t - 1]
            df1_r = df1 * self.f1_cache[t] * (1.0 - self.f1_cache[t])

            dWf1 += np.dot(df1_r, concat1.T)
            dWi1 += np.dot(di1_r, concat1.T)
            dWc1 += np.dot(dc1t_r, concat1.T)
            dWo1 += np.dot(do1_r, concat1.T)
            dbf1 += df1_r; dbi1 += di1_r; dbc1 += dc1t_r; dbo1 += do1_r

            dcat1 = (np.dot(self.Wf1.T, df1_r) + np.dot(self.Wi1.T, di1_r) +
                     np.dot(self.Wc1.T, dc1t_r) + np.dot(self.Wo1.T, do1_r))

            dh1_next = dcat1[self.input_size:, :]
            dc1_next = self.f1_cache[t] * dc1

        # Adam update for all parameters
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        grad_map = {
            'Wf1': dWf1, 'Wi1': dWi1, 'Wc1': dWc1, 'Wo1': dWo1,
            'bf1': dbf1, 'bi1': dbi1, 'bc1': dbc1, 'bo1': dbo1,
            'Wf2': dWf2, 'Wi2': dWi2, 'Wc2': dWc2, 'Wo2': dWo2,
            'bf2': dbf2, 'bi2': dbi2, 'bc2': dbc2, 'bo2': dbo2,
            'Wy': dWy, 'by': dby
        }

        for name in self.param_names:
            grad = grad_map[name]
            np.clip(grad, -5.0, 5.0, out=grad)

            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad ** 2)

            m_hat = self.m[name] / (1.0 - beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - beta2 ** self.t)

            w = getattr(self, name)
            l2_term = l2_lambda * w if name.startswith('W') else 0.0
            setattr(self, name, w - lr * (m_hat / (np.sqrt(v_hat) + eps) + l2_term))

    def predict(self, x_sequence):
        T = x_sequence.shape[0]

        # Layer 1 forward - collect all h1[t]
        h1 = np.zeros((self.hidden_size, 1))
        c1 = np.zeros((self.hidden_size, 1))
        h1_all = []

        for t in range(T):
            xt = x_sequence[t].reshape(-1, 1)
            concat1 = np.vstack((xt, h1))

            f1 = self.sigmoid(np.dot(self.Wf1, concat1) + self.bf1)
            i1 = self.sigmoid(np.dot(self.Wi1, concat1) + self.bi1)
            ct1 = np.tanh(np.dot(self.Wc1, concat1) + self.bc1)
            o1 = self.sigmoid(np.dot(self.Wo1, concat1) + self.bo1)

            c1 = f1 * c1 + i1 * ct1
            h1 = o1 * np.tanh(c1)
            h1_all.append(h1.copy())

        # Layer 2 forward
        h2 = np.zeros((self.hidden_size, 1))
        c2 = np.zeros((self.hidden_size, 1))

        for t in range(T):
            concat2 = np.vstack((h1_all[t], h2))

            f2 = self.sigmoid(np.dot(self.Wf2, concat2) + self.bf2)
            i2 = self.sigmoid(np.dot(self.Wi2, concat2) + self.bi2)
            ct2 = np.tanh(np.dot(self.Wc2, concat2) + self.bc2)
            o2 = self.sigmoid(np.dot(self.Wo2, concat2) + self.bo2)

            c2 = f2 * c2 + i2 * ct2
            h2 = o2 * np.tanh(c2)

        z = np.dot(self.Wy, h2) + self.by
        return self.stable_softmax(z)
