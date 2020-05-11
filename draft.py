import os
import numpy as np
from collections import OrderedDict
from itertools import chain
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd

#np.random.seed(3)

os.chdir('/home/firedragon/Desktop/ACADEMIC/DD2424/A4/')

harry_book = 'goblet_book.txt'
book_data = open(harry_book, "r").read()#.split('\n')
no_lines = len(book_data)
book_chars = list(OrderedDict.fromkeys(chain.from_iterable(book_data)))
K = len(book_chars)

book_alphabet_i = {i: book_chars[i] for i in range(K)}
book_alphabet_c = {book_chars[i]: i for i in range(K)}

book_data_join = ''
for i in range(len(book_data)):
    book_data_join += book_data[i]

print(len(book_data_join))

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
    theta -= (eta/np.sqrt(m + eps)) * g
    return theta, m

def CrossEntropy(y, p):
    cross_entropy_loss = -np.log(np.multiply(y, p).sum(axis=0))
    loss = np.sum(cross_entropy_loss)
    return loss

def Clipping(RNN_U, RNN_W, RNN_V, RNN_b, RNN_c):
    return np.clip(RNN_U, -5, 5), np.clip(RNN_W, -5, 5), np.clip(RNN_V, -5, 5), np.clip(RNN_b, -5, 5), np.clip(RNN_c, -5, 5)

m = 100

sig = .01
eta = .1
seq_length = 25
book = book_data_join

h0 = np.zeros((m, 1))
h = np.zeros((m, 1))

#RNN_b = np.zeros((m, 1))
#RNN_c = np.zeros((K, 1))

RNN_b = np.random.normal(0, sig, (m, 1))
RNN_c = np.random.normal(0, sig, (K, 1))

# np.random.seed(4)

RNN_U = np.random.normal(0, sig, (m, K))
RNN_W = np.random.normal(0, sig, (m, m))
RNN_V = np.random.normal(0, sig, (K, m))

m_V = np.zeros((K, m))
m_W = np.zeros((m, m))
m_U = np.zeros((m, K))

m_b = np.zeros((m, 1))
m_c = np.zeros((K, 1))

dL_db = np.zeros((m, 1))
dL_dc = np.zeros((K, 1))

dL_dU = np.zeros((m, K))
dL_dW = np.zeros((m, m))
dL_dV = np.zeros((K, m))


def Fwd(x, h, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c):
    x = x.reshape(K, -1)
    P = np.empty((K, x.shape[1]))
    O = np.empty((K, x.shape[1]))
    H = np.empty((m, x.shape[1]))
    A = np.empty((m, x.shape[1]))

    ht = np.copy(h)
    '''
    for t in range(x.shape[1]):
        # print(x.shape)
        # print(ht.shape)

        A[:, t] = RNN_W @ ht[:, 0] + RNN_U @ x[:, t] + RNN_b[:, 0]  # mx1
        ht = np.tanh(A[:, t].reshape(-1, 1))  # mx1
        H[:, t] = np.copy(ht[:, 0])
        O[:, t] = RNN_V @ ht[:, 0] + RNN_c[:, 0]  # Cx1
        P[:, t] = softmax(O[:, t])  # Cx1
    '''

    for t in range(x.shape[1]):
        A[:, t] = (np.dot(RNN_W, ht.reshape(-1, 1)) + np.dot(RNN_U, x[:, t]).reshape(-1,1) + RNN_b).reshape(-1)
        ht = np.tanh(A[:, t])
        H[:, t] = np.copy(ht)
        O[:, t] = (np.dot(RNN_V, ht.reshape(-1, 1)) + RNN_c).reshape(-1)
        P[:, t] = softmax(O[:, t])

    return A, H, O, P

def NumGrads(x, y, w_name, h, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c, dL_dW):
    hprev = np.zeros((dL_dW.shape[0], 1))
    grad_num = np.zeros(RNN_V.shape)
    print("runing numerically gradients\n")

    for i in range(RNN_V.shape[0]):
        for j in range(RNN_V.shape[1]):
            RNN_V[i][j] -= h
            p = Fwd(x, hprev, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c)[3]
            l1 = CrossEntropy(y, p)
            RNN_V[i][j] += h * 2
            p = Fwd(x, hprev, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c)[3]
            l2 = CrossEntropy(y, p)
            RNN_V[i][j] -= h
            grad_num[i][j] = (l2 - l1) / (2 * h)
    return grad_num

def Compute(X, Y, P, H, dL_dV, dL_dU, dL_dW, dL_db, dL_dc, RNN_W):

    h_next = np.zeros_like(H[:, 0].reshape(-1, 1))
    for t in reversed(range(seq_length)):
        dL_do = P[:, t].reshape(-1, 1) - Y[:, t].reshape(-1, 1)
        ht = H[:, t].reshape(-1, 1)
        dL_dV += np.dot(dL_do, ht.T)
        dL_dc += dL_do
        dL_dh = np.dot(RNN_V.T, dL_do) + h_next
        dL_da = np.multiply(dL_dh, (1 - np.square(ht)))
        dL_dU += np.dot(dL_da, X[:, t].reshape(1, -1))
        dL_dW += np.dot(dL_da, H[:, t-1].reshape(1, -1))
        dL_db += dL_da
        h_next = RNN_W.T @ dL_da

    return dL_dV, dL_dU, dL_dW, dL_db, dL_dc

def Bwd(RNN_V, RNN_W, RNN_U, RNN_b, RNN_c, g_V, g_W, g_U, g_b, g_c, m_V, m_W, m_U, m_b, m_c, eta):
    RNN_V, m_V = AdaGrad(g = g_V, m = m_V, theta = RNN_V, eta = eta, eps = np.finfo(float).eps)
    RNN_W, m_W = AdaGrad(g = g_W, m = m_W, theta = RNN_W, eta = eta, eps = np.finfo(float).eps)
    RNN_U, m_U = AdaGrad(g = g_U, m = m_U, theta = RNN_U, eta = eta, eps = np.finfo(float).eps)
    RNN_b, m_b = AdaGrad(g = g_b, m = m_b, theta = RNN_b, eta = eta, eps = np.finfo(float).eps)
    RNN_c, m_c = AdaGrad(g = g_c, m = m_c, theta = RNN_c, eta = eta, eps = np.finfo(float).eps)

    #RNN_V -= 0.000001*g_V
    #RNN_W -= 0.000001*g_W
    #RNN_U -= 0.000001*g_U
    #RNN_b -= 0.000001*g_b
    #RNN_c -= 0.000001*g_c

    return RNN_V, RNN_W, RNN_U, RNN_b, RNN_c, m_V, m_W, m_U, m_b, m_c

def Synth(x, h, n, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c):
    Y = np.zeros((K, n))
    for j in range(n):

        _, hnext, _, pnext = Fwd(x, h, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c)
        cp = pd.Series(pnext[:, 0]).cumsum()
        ixs = np.where(cp - np.random.rand() > 0)[0][0]
        Y[ixs, j] = 1
        xnext = Y[:, j].reshape(-1, 1)

        x = xnext[:]
        h = hnext[:]

    return ind_to_char(Y)

'''
seq_length = 25
X_chars = book_data_join[0 : seq_length]
#print('X: ', X_chars)
Y_chars = book_data_join[1 : seq_length + 1]
#print('Y: ', Y_chars)
X = char_to_ind(X_chars) # K x seq_length
Y = char_to_ind(Y_chars) # K x seq_length

### TEST Gradients
num_V = NumGrads(X, Y, 'RNN_V', 1e-04, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c, dL_dW)
_, H, _, P = Fwd(X, h0, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c)
dL_dV, dL_dU, dL_dW, dL_db, dL_dc = Compute(X, Y, P, H, dL_dV, dL_dU, dL_dW, dL_db, dL_dc, RNN_W)

aux1 = np.abs(num_V - dL_dV)
aux2 = np.abs(num_V) + np.abs(dL_dV)
checking_criteria = np.max(aux1/np.maximum(1e-04, aux2))
print(checking_criteria)
#print('')
#print(num_V)
#print('')
#print(dL_dV)
'''

epochs = 7

smooth_loss = 0

n_sentence = 300

loss_rec = []
smooth_loss_rec = []

sentences = []

iter = 0

print(len(book_data_join))

best_loss = 50

for ep in range(epochs):
    hprev = np.zeros((m, 1))
    for e in range(0, len(book_data_join)-seq_length, seq_length):
        #print('\nWhere: ', e)
        X_chars = book_data_join[e : e + seq_length]
        Y_chars = book_data_join[e + 1 : e + seq_length + 1]

        #print('e', e)
        #print(X_chars)
        #print(Y_chars)

        X = char_to_ind(X_chars)
        Y = char_to_ind(Y_chars)

        _, H, _, P = Fwd(X[:], hprev[:],
                         RNN_V[:], RNN_U[:], RNN_W[:], RNN_b[:], RNN_c[:])

        dL_dV, dL_dU, dL_dW, dL_db, dL_dc = Compute(X[:], Y[:], P[:], H[:],
                                                    dL_dV[:], dL_dU[:], dL_dW[:], dL_db[:], dL_dc[:],
                                                    RNN_W[:])

        hprev = np.copy(H[:, -1].reshape(-1, 1))
        #print(m_b)
        RNN_V, RNN_W, RNN_U, RNN_b, RNN_c, m_V, m_W, m_U, m_b, m_c = Bwd(RNN_V[:], RNN_W[:], RNN_U[:], RNN_b[:], RNN_c[:],
                                                                         dL_dV[:], dL_dW[:], dL_dU[:], dL_db[:], dL_dc[:],
                                                                         m_V[:], m_W[:], m_U[:], m_b[:], m_c[:], eta)
        #print(m_b)
        RNN_U, RNN_W, RNN_V, RNN_b, RNN_c = Clipping(RNN_U[:], RNN_W[:], RNN_V[:], RNN_b[:], RNN_c[:])

        dL_db = np.zeros((m, 1))
        dL_dc = np.zeros((K, 1))

        dL_dU = np.zeros((m, K))
        dL_dW = np.zeros((m, m))
        dL_dV = np.zeros((K, m))

        loss = CrossEntropy(Y, P)
        smooth_loss = loss if smooth_loss == 0 else 0.999 * smooth_loss + 0.001 * loss
        loss_rec.append(loss)

        if iter % 100 == 0:
            print('')
            print("Interation", iter, "Smooth loss: ", smooth_loss)
            smooth_loss_rec.append(smooth_loss)

        if iter % 500 == 0:
            sentence = Synth(X[:,0], hprev, n_sentence, RNN_V, RNN_U, RNN_W, RNN_b, RNN_c)
            sentences.append(sentence)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_model = [RNN_U, RNN_W, RNN_V, RNN_b, RNN_c]

        iter += 1


for i, sentence in enumerate(sentences):
    print("Iteraction: ", i*500)
    print(sentence)
    print("")

plt.figure()
plt.plot(loss_rec)
plt.title('Loss')
plt.show()

plt.figure()
plt.plot(smooth_loss_rec)
plt.title('Smoothed Loss')
plt.show()

best_sentence = Synth(char_to_ind(book_data_join[0 : 0 + seq_length]), np.zeros((m, 1)), 10000, best_model[0], best_model[1], best_model[2], best_model[3], best_model[4])

print('Best Writing')
print(best_sentence)


