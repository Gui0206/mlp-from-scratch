"""
network.py — o MLP generalizado para um número ARBITRÁRIO de camadas.

Diferença para o testes.ipynb:
  - Lá os pesos eram fixos (W1, W2). Aqui eles viram uma LISTA (self.weights),
    e o forward/backward viram LOOPS sobre as camadas.
  - As camadas OCULTAS usam a ativação configurável (ReLU); a SAÍDA usa softmax.

A arquitetura é descrita por uma lista de tamanhos, por exemplo:
    [784, 128, 64, 10]  ->  entrada 784, duas ocultas (128 e 64), saída 10

Estratégia recomendada: preencha os TODOs e teste PRIMEIRO no XOR (lá no fim do
arquivo, em __main__). Se a rede aprender o XOR, seus gradientes estão certos e
você pode escalar para o MNIST com confiança.
"""
import numpy as np

from mlp.activations import (
    relu, relu_derivative,
    sigmoid, sigmoid_derivative,
    softmax,
)
from mlp.losses import (
    categorical_cross_entropy,
    categorical_cross_entropy_gradient,
)


# Mapeia o NOME da ativação -> (função, derivada). Usado só nas camadas OCULTAS.
# É isto que torna a ativação "configurável" (requisito da atividade) e permite
# comparar ReLU x sigmoid depois.
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
}


class MLP:
    def __init__(self, layer_sizes, optimizer, activation="relu"):
        """
        layer_sizes: lista [n_entrada, n_oculta1, ..., n_saida]
        optimizer:   instância de SGD (é ela quem guarda o learning_rate)
        activation:  ativação das camadas OCULTAS ("relu" ou "sigmoid")
        """
        self.layer_sizes = layer_sizes
        self.optimizer = optimizer
        self.activation, self.activation_derivative = ACTIVATIONS[activation]

        # ---- cria os parâmetros de cada camada ----
        # Há (len(layer_sizes) - 1) camadas de pesos.
        # Ex.: [784,128,64,10] -> shapes (784,128), (128,64), (64,10).
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            #  matriz de pesos com shape (n_in, n_out).
            #   NÃO use zeros: todos os neurônios ficariam simétricos e
            #   aprenderiam exatamente igual (a rede não sairia do lugar).
            #   Dica (He init, boa para ReLU):
            #       np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

            # bias pode começar em zero, shape (1, n_out).
            #   Dica: np.zeros((1, n_out))
            b = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Passa X pela rede e GUARDA os valores intermediários (z e a), porque o
        backward vai precisar deles.

          z_values[i] = a_values[i] @ W[i] + b[i]   (pré-ativação da camada i)
          a_values[i+1] = ativação(z_values[i])     (saída da camada i)
        """
        self.z_values = []      # pré-ativações de cada camada
        self.a_values = [X]     # ativações; a_values[0] é a própria entrada X

        n_layers = len(self.weights)
        a = X
        for i in range(n_layers):
            # TODO: z = a @ W[i] + b[i]
            #   Dica: np.dot(a, self.weights[i]) + self.biases[i]
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == n_layers - 1:
                # ÚLTIMA camada -> softmax (transforma em probabilidades)
                a = softmax(z)
            else:
                # camadas OCULTAS -> ativação configurável (ReLU)
                # TODO: a = self.activation(z)
                a = self.activation(z)

            self.a_values.append(a)

        return a

    # ------------------------------------------------------------------
    def backward(self, X, y_true):
        """
        Backpropagation pelas camadas, de TRÁS para FRENTE.
        y_true: rótulos em one-hot, shape (batch, n_saida).
        """
        m = X.shape[0]                  # tamanho do batch
        n_layers = len(self.weights)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        # ---- delta da camada de SAÍDA ----
        # softmax + cross-entropy => gradiente nos logits = (saída - y).
        # Dividimos por m para virar a MÉDIA do batch (deixa o learning rate
        # independente do tamanho do batch).
        delta = categorical_cross_entropy_gradient(y_true, self.a_values[-1]) / m

        # ---- loop de trás para frente ----
        for i in reversed(range(n_layers)):
            # gradiente dos pesos desta camada: (entrada da camada).T @ delta
            # TODO: grad_W[i] = np.dot(self.a_values[i].T, delta)
            grad_W[i] = np.dot(self.a_values[i].T, delta)

            # gradiente do bias: soma do delta ao longo do batch
            # TODO: grad_b[i] = np.sum(delta, axis=0, keepdims=True)
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)

            # propaga o delta para a camada ANTERIOR (se não for a primeira).
            # Aqui usamos a derivada da ATIVAÇÃO OCULTA em z_values[i-1].
            if i > 0:
                # TODO:
                #   delta = np.dot(delta, self.weights[i].T) \
                #           * self.activation_derivative(self.z_values[i - 1])
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

        # ---- monta listas alinhadas [W0,b0,W1,b1,...] e deixa o optimizer aplicar ----
        params, grads = [], []
        for i in range(n_layers):
            params.append(self.weights[i])
            params.append(self.biases[i])
            grads.append(grad_W[i])
            grads.append(grad_b[i])
        self.optimizer.step(params, grads)

    # ------------------------------------------------------------------
    def fit(self, X, y, n_epochs=20, batch_size=64, verbose=True):
        """
        Treina por MINI-BATCHES.
          X: (n_amostras, n_entrada)     y: one-hot (n_amostras, n_saida)
        Retorna um histórico de loss/acurácia por época (para os plots).
        """
        n = X.shape[0]
        history = {"loss": [], "acc": []}

        for epoch in range(n_epochs):
            # 1) embaralhar a cada época (o "stochastic" do SGD)
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]

            # 2) percorrer os mini-batches
            for start in range(0, n, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]
                # um passo de treino = forward seguido de backward
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # 3) registrar métricas da época (na base inteira, só p/ acompanhar)
            output = self.forward(X)
            loss = categorical_cross_entropy(y, output)
            acc = self.accuracy(X, y)
            history["loss"].append(loss)
            history["acc"].append(acc)
            if verbose:
                print(f"Época {epoch + 1:3d}/{n_epochs} - loss: {loss:.4f} - acc: {acc:.4f}")

        return history

    # ------------------------------------------------------------------
    def predict(self, X):
        """Retorna a CLASSE prevista (índice 0..9) para cada amostra."""
        output = self.forward(X)
        # TODO: a classe prevista é o índice de maior probabilidade em cada linha
        return np.argmax(output, axis=1)

    def accuracy(self, X, y):
        """Fração de acertos. y em one-hot."""
        preds = self.predict(X)
        # TODO: compare a previsão com a classe verdadeira (argmax de y):
        #   
        return np.mean(preds == np.argmax(y, axis=1))


# ----------------------------------------------------------------------
# SMOKE TEST no XOR — rode depois de preencher os TODOs:
#     python -m mlp.network
# Se sair [0 1 1 0], a estrutura (forward + backprop) está correta e você
# pode escalar para o MNIST. (Os hiperparâmetros a gente ajusta junto se
# não convergir de primeira.)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from mlp.optimizers import SGD

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    # one-hot de 2 classes: 0 -> [1,0], 1 -> [0,1]
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)

    net = MLP([2, 16, 2], optimizer=SGD(learning_rate=0.5), activation="relu")
    net.fit(X, y, n_epochs=2000, batch_size=4, verbose=False)

    print("Predições XOR:", net.predict(X), " (esperado: [0 1 1 0])")
    print("Acurácia:", net.accuracy(X, y))
