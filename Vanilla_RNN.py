import os
import numpy as np
from collections import OrderedDict
from itertools import chain
from scipy.special import softmax
import pandas as pd

os.chdir('/home/firedragon/Desktop/ACADEMIC/DD2424/A4/')

harry_book = 'goblet_book.txt'
book_data = open(harry_book, "r").read().split('\n')
no_lines = len(book_data)
book_chars = list(OrderedDict.fromkeys(chain.from_iterable(book_data)))
K = len(book_chars)

book_alphabet_i = {i: book_chars[i] for i in range(K)}
book_alphabet_c = {book_chars[i]: i for i in range(K)}

book_data_join = ''
for i in range(len(book_data)):
    book_data_join += book_data[i]

def char_to_ind(char, alphabet=book_alphabet_c):
    idxs = [alphabet[ch] for ch in char]
    ind = np.zeros((len(alphabet), len(idxs)))
    for i, elem in enumerate(idxs):
        ind[elem, i] = 1
    return ind

def ind_to_char(ind, alphabet=book_alphabet_i):
    n_cols = ind.shape[1]
    char = ''
    for c in range(n_cols):
        i = np.argmax(ind[:, c])
        char = char + alphabet[i]
    return char

def AdaGrad(g, m, theta, eta, eps):
    m += g**2
    theta += - eta/np.sqrt(m + eps) @ g
    return theta

def CrossEntropy(y, p):
    loss = -np.sum(np.log(y.T @ p)) # sum over tau
    return loss

class RNN:

    def __init__(self, m = 100,
                 K = K, eta = .1,
                 seq_length = 25, sig = .01,
                 book_data = book_data_join):

        self.K = K
        self.m = m

        self.eta = eta
        self.seq_length = seq_length
        self.book = book_data

        self.h0 = np.zeros((100, 1))
        self.h = np.zeros((100, 1))

        self.RNN_b = np.zeros((m, 1))
        self.RNN_c = np.zeros((K, 1))

        self.RNN_U = sig * np.random.normal(0, 1, m * K).reshape((m, K))
        self.RNN_W = sig * np.random.normal(0, 1, m * m).reshape((m, m))
        self.RNN_V = sig * np.random.normal(0, 1, K * m).reshape((K, m))

        self.m_V = np.zeros((K, m))
        self.m_W = np.zeros((m, m))
        self.m_U = np.zeros((m, K))

        self.m_b = np.zeros((m, 1))
        self.m_c = np.zeros((K, 1))

        #self.dL_db = np.zeros((m, 1))
        #self.dL_dc = np.zeros((K, 1))

        #self.dL_dU = np.zeros((m, K))
        #self.dL_dW = np.zeros((m, m))
        #self.dL_dV = np.zeros((K, m))

    def Synth(self, x, h, n):
        Y = np.zeros((self.K, n))
        for j in range(n):
            _, hnext, _, pnext = self.Fwd(x, h)
            cp = pd.Series(pnext[:, 0]).cumsum()
            ixs = np.where(cp - np.random.rand() > 0)[0][0]
            Y[ixs, j] = 1
            xnext = Y[:, j].reshape(-1, 1)

            x = xnext[:]
            h = hnext[:]
        return ind_to_char(Y)

    def Fwd(self, x, h):
        a = self.RNN_W @ h + self.RNN_U @ x + self.RNN_b  # mx1
        h = np.tanh(a)  # mx1
        o = self.RNN_V @ h + self.RNN_c  # Cx1
        p = softmax(o)  # Cx1
        return a, h, o, p

    def Bwd(self, g_V, g_W, g_U):
        self.RNN_V = AdaGrad(g = g_V, m = self.m_V, theta = self.RNN_V, eta = self.eta, eps = 0.0001)
        self.RNN_W = AdaGrad(g = g_W, m = self.m_W, theta = self.RNN_W, eta = self.eta, eps = 0.0001)
        self.RNN_U = AdaGrad(g = g_U, m = self.m_U, theta = self.RNN_U, eta = self.eta, eps = 0.0001)
        #self.RNN_b = AdaGrad(g = g_b, m = self.m_b, theta = self.RNN_b, eta = self.eta, eps = 0.0001)
        #self.RNN_c = AdaGrad(g = g_c, m = self.m_c, theta = self.RNN_c, eta = self.eta, eps = 0.0001)


class Gradients(RNN):
    def __init__(self, m = 100, K = K):

        RNN.__init__(self, m = m, K = K)

        #self.K = K
        #self.m = m

        self.dL_db = np.zeros((self.m, 1))
        self.dL_dc = np.zeros((self.K, 1))

        self.dL_dU = np.zeros((self.m, self.K))
        self.dL_dW = np.zeros((self.m, self.m))
        self.dL_dV = np.zeros((self.K, self.m))

    def NumGrads(self, X, Y, grad_name, hh):

        #n = getattr(Gradients, grad_name).shape[0] * getattr(Gradients, grad_name).shape[1]
        #num_grad = np.zeros(getattr(Gradients, grad_name).shape)
        num_grad = np.zeros(getattr(self, grad_name).shape)
        hprev = np.zeros((self.dL_dW.shape[0], 1))

        grad_try = np.array(getattr(self, grad_name))

        l = []
        for i in range(getattr(self, grad_name).shape[0]):
            for j in range(getattr(self, grad_name).shape[1]):
                for k in [-1,1]:
                    grad_try[i, j] = getattr(self, grad_name)[i, j] + k*hh
                    setattr(self, grad_name, grad_try)
                    a, h, o, p = self.Fwd(X, hprev)
                    l.append(CrossEntropy(Y, p))
                num_grad[i, j] = (l[1] - l[0]) / hh
        return num_grad

    def Compute(self, y, tau, x, h0):

        #self.dL_dV = np.zeros((self.K, self.m))
        #self.dL_dU = np.zeros((self.m, self.K))
        #self.dL_dW = np.zeros((self.m, self.m))

        h_all = [h0]
        a_all = []
        for t in range(tau):
            a_t, h_t, _, p_t = self.Fwd(x[:, t], h_all[t])
            a_all.append(a_t)
            h_all.append(h_t)
            dL_do = -(y[:, t] - p_t).T
            self.dL_dV += dL_do.T @ h_t.T

        dL_dh = dL_do @ self.RNN_V
        for t in range(tau-1, 1):
            dL_da = dL_dh @ np.diag(1 - np.tanh(a_all[t+1]))
            self.dL_dW += dL_da.T @ h_all[t].T
            self.dL_dU += dL_da.T @ x[:, t].T
            dL_dh = dL_do @ self.RNN_V + dL_da @ self.RNN_W


model = RNN()
x0_word = 'x'
x0_word_i = char_to_ind(x0_word)
h0 = np.random.rand(100).reshape(-1,1)
gen = model.Synth(x0_word_i, h0, 10)
print(gen)

#seq_length = 25
#X_chars = book_data_join[0 : seq_length]
#print('X: ', X_chars)
#Y_chars = book_data_join[1 : seq_length + 1]
#print('Y: ', Y_chars)
#X = char_to_ind(X_chars) # K x seq_length
#Y = char_to_ind(Y_chars) # K x seq_length
#model = RNN()
#Grads = Gradients()
#h0 = np.zeros((100, 1))

### TEST Gradients
#num_V = Grads.NumGrads(X, Y, 'dL_dV', 1e-06)
#Grads.Compute(Y, seq_length, X, h0)

#print(num_V - Grads.dL_dV)

#a, h, o, p = model.Fwd(X, h0)
#model.Bwd(g_V, g_W, g_U)
#print('P: ', ind_to_char(p))



