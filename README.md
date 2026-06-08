Dia 01:

Primeiramente eu consegui implementar um perceptron simples na pasta de testes.

Dia 02:

Ainda na pasta de testes, para o MLP, eu pedi para o claude me dar uma estrutura de código pré pronto para que eu preencha, vou deixar parte do código que ele me deu aqui:

```
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


```

Dia 03 parte 1:

Agora é só modularizar isso para os arquivos do projeto.

Acabei de perceber que eu havia implementado a sigmoid, mas ela não vai funcionar para o MNSIT por que a derivada dela é no máximo 0.25. O Claude já me ajudou me dando o protótipo da Relu no arquivo activations.py para implementar.

Depois de uns vídeos no youtube eu entendi o que a SoftMax faz; Pega os resultados da camada de saída e transforma em probabilidades que somam 1.

Dia 03 parte 2:

Acabei de implementar o SGD. O conceito de optimizer estava um pouco confuso, não entendia a diferença com a LOSS. Mas o optimizer é o algoritmo que calcular o reajuste a partir da métrica LOSS.