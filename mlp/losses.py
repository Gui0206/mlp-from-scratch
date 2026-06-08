"""
Funções de loss (custo) e seus gradientes.

Para classificação multiclasse usamos a CATEGORICAL cross-entropy, que trabalha
com a saída da softmax (probabilidades) e os rótulos em formato one-hot.

  one-hot: o dígito 3, com 10 classes, vira  [0,0,0,1,0,0,0,0,0,0]
"""
import numpy as np


def categorical_cross_entropy(y_true, y_pred):
    """
    Valor da loss (um escalar), usado só para ACOMPANHAR o treino.

    y_true: one-hot,            shape (batch, n_classes)
    y_pred: saída da softmax,   shape (batch, n_classes)

    Fórmula:  -mean( soma_por_linha( y_true * log(y_pred) ) )
    Como y_true é one-hot, em cada linha só "sobra" o log da classe correta.
    """
    # clip evita log(0), que seria -infinito
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    # por linha sobra só o log da classe correta; tira a média e inverte o sinal
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_gradient(y_true, y_pred):
    """
    Gradiente da loss em relação aos LOGITS (a entrada z da última camada,
    ANTES da softmax).

    Aqui acontece de novo o cancelamento elegante que você viu com
    sigmoid + binary cross-entropy: a derivada da softmax cancela com o
    denominador da cross-entropy, e o gradiente vira simplesmente:

        delta_saida = y_pred - y_true

    Ou seja, no network.py você NÃO precisa derivar a softmax na mão —
    é só usar este resultado direto como o "delta" da camada de saída.
    """
    return y_pred - y_true
