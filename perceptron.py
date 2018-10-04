#https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b
#sudo apt-get install python3-pip
#sudo pip install virtualenv 
#source venv/bin/activate

#se nao der:
#virtualenv -p /usr/bin/python3 venv
#source venv/bin/activate
#pip install pandas
#pip install numpy
#pip install matplotlib
import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self, file_name, N):
        self.df = pd.read_csv(file_name, header=None)
        self.A =  len(self.df.columns) - 1 # Atributos
        self.N = N # Neuronios
        #self.M = len(self.df)
        #print("Tabela completa: ")
        #print(self.df)
        #print("Tabela embaralhada: ")
        self.embaralhar()
        #print(self.df)
        #print("Separar saida da entrada: ")
        self.separar_entrada_saida()
        #print(self.entrada)
        #print(self.saida)
        #print("Separar conjunto de teste e conjunto de treino")
        self.separar_conjuntos()
        #print("Teste:")
        #print(self.conjunto_teste_entrada)
        #print(self.conjunto_teste_saida)
        #print("Treino:")
        #print(self.conjunto_treino_entrada)
        #print(self.conjunto_treino_saida)
        self.perceptron()

    def embaralhar(self):
        self.df = self.df.sample(frac=1)  #embaralha o dataframe

    def separar_entrada_saida(self):
        self.entrada = pd.DataFrame(self.df[self.df.columns[-self.A:]].values) #entradas
        self.saida = self.df[self.df.columns[0]] #saidas
        self.saida = pd.DataFrame(self.saida.apply(self.transformar).values.tolist()) #saidas transformadas e ordenadas

    def transformar(self, df):
        if (df == 1):
            return [1, 0, 0]

        if (df == 2):
            return [0, 1, 0]

        if (df == 3):
            return [0, 0, 1]


    def separar_conjuntos(self):
        tam = len(self.entrada) #177
        length_slice = int(round(tam / 3))  #entrada / 3 = 59
        self.T = length_slice #Numero de itens para teste 59
        self.M = tam - length_slice #Numero de itens para treinamento 177-59
        print("Items: ")
        print("   Teste: " + str(self.T))
        print("   Treino: " + str(self.M))

        self.conjunto_teste_entrada = self.entrada[:length_slice]  #primeiros length_slice da entrada
        self.conjunto_teste_saida = self.saida[:length_slice]  #primeiros length_slice da saida
        
        self.conjunto_treino_entrada = self.entrada[length_slice:] #ultimos length_slice da entrada
        self.conjunto_treino_saida = self.saida[length_slice:] #ultimos length_slice da entrada

    def degrau(self, df):
        if (df >= 0):
            return 1
        return 0

    def perceptron(self):
        #import pdb; pdb.set_trace()
        max_it = 200
        t = 0  # Iteracao (Epoca)
        b = np.zeros(self.N)  # Bias  vetor vazio com N zeros
        W = pd.DataFrame(np.zeros((self.N, self.A))) # dataframe N-A preenchidos com 0
        y = pd.DataFrame(np.zeros(self.N))  #N zeros
        e = pd.DataFrame(np.zeros(self.N))  #N zeros
        d = self.conjunto_treino_saida
        E = pd.DataFrame(np.zeros(max_it))
        alfa = 0.3
        corretos = 0       

        while (t < max_it):    
            for i in range(self.M):
                y = W.dot(self.conjunto_treino_entrada.iloc[i]).add(b).apply(self.degrau)
                e = d.iloc[i].subtract(y) # determinando erro
                E.iloc[t] += e.dot(e.T)
                update = e.values.reshape(self.N,1).dot(self.conjunto_treino_entrada.iloc[[i]]) * alfa
                W = W + update
                b = b + alfa * e
                #print(b)

            t += 1
                
        for i in range(self.T):
            y = W.dot(self.conjunto_teste_entrada.iloc[i]).add(b).apply(self.degrau)
            if y.equals(self.conjunto_teste_saida.iloc[i]):
                corretos += 1

        print("Corretude: " + str((float(corretos)/self.T)*100))
                    

p = Perceptron('wine.data', 3)

