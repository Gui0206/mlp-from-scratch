"""
Funções de ativação e suas derivadas.

Cada ativação tem duas funções: a própria (usada no forward) e a derivada
(usada no backward). A softmax é um caso especial — veja o comentário lá embaixo.
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# ---------------------------------------------------------------------------
# ReLU: f(x) = max(0, x).
# É a ativação padrão das camadas OCULTAS hoje em dia (não satura como a
# sigmoid, e o gradiente não "some" em redes profundas). Use ReLU no MNIST.
# ---------------------------------------------------------------------------
def relu(x):
    # TODO: retorne x onde x > 0, e 0 caso contrário.
    # Dica: np.maximum(0, x)
    return np.maximum(0,x)


def relu_derivative(x):
    # A derivada da ReLU é 1 onde x > 0 e 0 onde x <= 0.
    return np.where(x > 0, 1, 0)


# ---------------------------------------------------------------------------
# Softmax: transforma um vetor de "scores" (logits) numa distribuição de
# probabilidade que soma 1. É a ativação da CAMADA DE SAÍDA na classificação
# multiclasse (os 10 dígitos).
#
#   softmax(x)_i = e^(x_i) / soma_j( e^(x_j) )
#
# Cuidado numérico: e^x estoura se x for grande. O truque é subtrair o máximo
# de cada linha ANTES de exponenciar — não muda o resultado, só estabiliza.
# (Não precisamos de softmax_derivative: no losses.py o gradiente da softmax
#  já vem combinado com a cross-entropy e simplifica para (pred - y).)
# ---------------------------------------------------------------------------
def softmax(x):
    shift = x - np.max(x, axis=1, keepdims=True)   # estabilidade
    exps  = np.exp(shift)
    return exps / np.sum(exps, axis=1, keepdims=True)