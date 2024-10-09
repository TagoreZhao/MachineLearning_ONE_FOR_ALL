import numpy as np

class RNN:
    def __init__(self, hidden_size, vocab_size, seq_len, learning_rate):
        # hyper parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_len
        self.learning_rate = learning_rate
        
        # model parameters
        self.U = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_size, vocab_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (vocab_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))
    
    def softmax(self, x):
        p = np.exp(x - np.max(x))
        return p / np.sum(p)
    
    def forward(self, inputs, hprev):
        xs, hs, os, ps = {}, {}, {}, {}
        #xs: input vectors at each time step
        #hs: hidden states at each time step
        #os: output vectors at each time step
        #ps: output probabilities at each time step
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.U, xs[t]) + np.dot(self.W, hs[t-1]) + self.b)
            os[t] = np.dot(self.V, hs[t]) + self.c
            ps[t] = self.softmax(os[t])
        return xs, hs, ps
    
    def loss(self, ps, targets):
        return sum(-np.log(ps[t][targets[t],0]) for t in range(self.seq_length))
    
    def backward(self, xs, hs, ycap, targets):
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ycap[t])
            dy[targets[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dc += dc
            dh = np.dot(self.V.T, dy) + dhnext
            dhrec = (1 - hs[t] * hs[t]) * dh
            db += dhrec
            dU += np.dot(dhrec, xs[t].T)
            dW += np.dot(self.W.T, dhrec)
            #pass the grdient from next cell for next iteration
            dhnext = np.dot(self.W.T, dhrec)
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out = dparam)
        return dU, dW, dV, db, dc
    
    def update_model(self, dU, dW, dV, db, dc):
        for param, dparam in zip([self.U, self.W, self.V, self.b, self.c],
                                 [dU, dW, dV, db,dc]):
            param += -self.learning_rate*dparam
            
    def predict(self, data_reader, start, n):
        x = np.zeros(self.vocab_size, 1)
        chars = [ch for ch in start]
        ixes = []
        for i in range(len(chars)):
            ix = data_reader.char_to_ix[chars[i]]
            x[ix] = 1
            ixes.append(ix)
        
        h = np.zeros((self.hidden_size, 1))
        
        for t in range(n):
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            y = np.dot(self.V, h) + self.c
            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p = p.ravel())
            x = np.zeros(self.vocab_size, 1)
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(data_reader.ix_to_char[i] for i in ixes)
        return txt        