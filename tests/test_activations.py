"""
Testes unitários das funções de ativação (mlp/activations.py).

Como rodar (a partir da raiz do projeto):
    pytest -v
    # ou, sem pytest:
    python tests/test_activations.py

A ideia de cada teste é comparar a saída da função com uma resposta que a
gente já sabe (casos conhecidos), checar propriedades que sempre valem
(invariantes) e garantir estabilidade numérica.
"""
import numpy as np

from mlp.activations import (
    relu,
    relu_derivative,
    softmax,
    sigmoid,
    sigmoid_derivative,
)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------
def test_relu_zera_negativos_e_mantem_positivos():
    x = np.array([-2.0, -0.5, 0.0, 3.0])
    assert np.array_equal(relu(x), [0.0, 0.0, 0.0, 3.0])


def test_relu_derivative_um_onde_positivo_zero_caso_contrario():
    x = np.array([-2.0, -0.5, 0.0, 3.0])
    # derivada: 1 onde x > 0, 0 onde x <= 0 (inclui o 0)
    assert np.array_equal(relu_derivative(x), [0, 0, 0, 1])


def test_relu_derivative_bate_com_gradiente_numerico():
    # Gradient check: compara a derivada analítica com a aproximação numérica
    # (f(x+eps) - f(x-eps)) / (2*eps). Evitamos x=0, onde a ReLU não é derivável.
    x = np.array([-3.0, -1.0, 0.5, 2.0, 5.0])
    eps = 1e-6
    numerico = (relu(x + eps) - relu(x - eps)) / (2 * eps)
    analitico = relu_derivative(x)
    assert np.allclose(numerico, analitico, atol=1e-5)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
def test_softmax_cada_linha_soma_um():
    logits = np.array([[1.0, 2.0, 3.0],
                       [4.0, 0.0, -1.0]])
    s = softmax(logits)
    assert np.allclose(s.sum(axis=1), [1.0, 1.0])


def test_softmax_entradas_iguais_viram_distribuicao_uniforme():
    logits = np.array([[1.0, 1.0, 1.0]])
    s = softmax(logits)
    assert np.allclose(s, [[1 / 3, 1 / 3, 1 / 3]])


def test_softmax_maior_logit_tem_maior_probabilidade():
    logits = np.array([[1.0, 2.0, 3.0]])
    s = softmax(logits)
    assert np.argmax(s[0]) == 2


def test_softmax_estavel_com_numeros_gigantes():
    # Sem o truque de subtrair o max, np.exp(1000) estouraria para inf.
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    s = softmax(logits)
    assert np.all(np.isfinite(s))
    assert np.isclose(s.sum(), 1.0)


def test_softmax_bate_com_a_formula_manual():
    logits = np.array([[1.0, 2.0, 3.0]])
    manual = np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()
    assert np.allclose(softmax(logits)[0], manual)


# ---------------------------------------------------------------------------
# Sigmoid (mantida como ativação alternativa para a comparação ReLU x sigmoid)
# ---------------------------------------------------------------------------
def test_sigmoid_em_zero_vale_meio():
    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)


def test_sigmoid_derivative_bate_com_gradiente_numerico():
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    eps = 1e-6
    numerico = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)
    analitico = sigmoid_derivative(x)
    assert np.allclose(numerico, analitico, atol=1e-5)


if __name__ == "__main__":
    # Permite rodar sem o pytest: python tests/test_activations.py
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
