Primeiramente eu consegui implementar um perceptron simples na pasta de testes.

Ainda na pasta de testes, para o MLP, eu pedi para o claude me dar uma estrutura de código pré pronto para que eu preencha, ele me deu isso:

```
import numpy as np

np.random.seed(42)

# === Funções de ativação ===
# Diferente do perceptron (step), aqui usamos uma ativação derivável,
# porque o backprop precisa da derivada.
def sigmoid(x):
    # 1 / (1 + e^(-x))
    pass

def sigmoid_derivative(x):
    # derivada da sigmoid. Dica: sig(x) * (1 - sig(x))
    pass


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, n_epochs=10000):
        # Camada 1: entrada -> escondida
        self.W1 =      # shape (n_inputs, n_hidden)
        self.b1 =      # shape (1, n_hidden)
        # Camada 2: escondida -> saída
        self.W2 =      # shape (n_hidden, n_outputs)
        self.b2 =      # shape (1, n_outputs)

        self.learning_rate =
        self.n_epochs =

    def forward(self, X):
        # Camada escondida
        self.z1 =      # X @ W1 + b1   (combinação linear)
        self.a1 =      # ativação(z1)
        # Camada de saída
        self.z2 =      # a1 @ W2 + b2
        self.a2 =      # ativação(z2)
        return self.a2

    def backward(self, X, y, output):
        # 1) Erro na saída
        erro_saida =          # (output - y)
        # 2) Gradiente na saída (erro * derivada da ativação em z2)
        delta_saida =

        # 3) Propaga o erro para a camada escondida
        erro_oculta =         # delta_saida @ W2.T
        # 4) Gradiente na camada escondida
        delta_oculta =        # erro_oculta * derivada da ativação em z1

        # 5) Atualiza pesos e bias (gradiente descendente)
        #    Dica: dW2 = a1.T @ delta_saida ; db2 = soma de delta_saida
        self.W2 -=
        self.b2 -=
        self.W1 -=
        self.b1 -=

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            # forward em todo o batch
            output =
            # backward (ajusta pesos)

            # (opcional) a cada N épocas, imprimir a loss para acompanhar
            # loss = np.mean((output - y) ** 2)

    def predict(self, X):
        # roda o forward e aplica o limiar: >= 0.5 -> 1, senão 0
        pass


# === Problema XOR (não é linearmente separável) ===
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])   # note o shape (4, 1)

mlp = MLP(n_inputs=2, n_hidden=4, n_outputs=1, learning_rate=0.1, n_epochs=10000)
mlp.fit(X, y)

print("Predições do MLP:")
print(mlp.predict(X))

```