#source venv/bin/activate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, file_name, N):
        self.df = pd.read_csv(file_name, header=None)
        self.A =  len(self.df.columns) - 1
        self.N = N
        self.embaralhar()
        self.separar_entrada_saida()
        self.separar_conjuntos()
        self.perceptron()

    def embaralhar(self):
        self.df = self.df.sample(frac=1)

    def separar_entrada_saida(self):
        self.entrada = pd.DataFrame(self.df[self.df.columns[-self.A:]].values)
        self.saida = self.df[self.df.columns[0]]
        self.saida = pd.DataFrame(self.saida.apply(self.transformar).values.tolist())

    def transformar(self, df):
        if (df == 3):
            return [0, 0, 1]

        if (df == 2):
            return [0, 1, 0]

        if (df == 1):
            return [1, 0, 0]

    def separar_conjuntos(self):
        tam = len(self.entrada)
        teste = int(round(tam / 3))
        self.T = teste
        self.M = tam - teste
        print('Test->' + str(self.T))
        print('Train->' + str(self.M))

        self.conjunto_teste_entrada = self.entrada[:teste]
        self.conjunto_teste_saida = self.saida[:teste]

        self.conjunto_treino_entrada = self.entrada[teste:]
        self.conjunto_treino_saida = self.saida[teste:]

    def degrau(self, df):
        if (df >= 0):
            return 1
        return 0

    def perceptron(self):
        #import pdb; pdb.set_trace()
        t = 0
        iteracoes = 250
        alpha = 0.3
        corretos = 0
        b = np.zeros(self.N)
        W = pd.DataFrame(np.zeros((self.N, self.A)))
        y = pd.DataFrame(np.zeros(self.N))
        e = pd.DataFrame(np.zeros(self.N))
        d = self.conjunto_treino_saida
        E = pd.DataFrame(np.zeros(iteracoes))

        while (t < iteracoes):
            for i in range(self.M):
                y = W.dot(self.conjunto_treino_entrada.iloc[i]).add(b).apply(self.degrau)
                e = d.iloc[i].subtract(y)
                E.iloc[t] += e.dot(e.T)
                update = e.values.reshape(self.N,1).dot(self.conjunto_treino_entrada.iloc[[i]]) * alpha
                W = W + update
                b = b + alpha * e

            t += 1

        for i in range(self.T):
            y = W.dot(self.conjunto_teste_entrada.iloc[i]).add(b).apply(self.degrau)
            if y.equals(self.conjunto_teste_saida.iloc[i]):
                corretos += 1

        print('Accuracy-> {}%'.format(str((float(corretos)/self.T)*100)))
        plt.plot(E)
        plt.show()

p = Perceptron('iris.data', 3)
