"""
Otimizadores: a regra de COMO atualizar os parâmetros a partir dos gradientes.

O network.py calcula os gradientes; o optimizer aplica a atualização.
É a mesma ideia da classe SGD que você fez no testes.ipynb.
"""
import numpy as np


class SGD:
    """Stochastic Gradient Descent:  param = param - learning_rate * grad."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        # params e grads são listas ALINHADAS (mesma ordem):
        #   params = [W1, b1, W2, b2, ...]   grads = [dW1, db1, dW2, db2, ...]
        for param, grad in zip(params, grads):

            param -= self.learning_rate * grad


# ---------------------------------------------------------------------------
# OPCIONAL (vale pontos na rubrica): um otimizador com momentum.
# Deixe para depois que o SGD básico estiver treinando o MNIST.
#
# class SGDMomentum:
#     def __init__(self, learning_rate=0.1, momentum=0.9):
#         ...
#     def step(self, params, grads):
#         # mantém uma "velocidade" por parâmetro:
#         #   v = momentum * v - lr * grad ; param += v
#         ...
# ---------------------------------------------------------------------------
