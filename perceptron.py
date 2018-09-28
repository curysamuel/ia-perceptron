from random import shuffle

def gera_vetor_amostra(vetor, qtd):
    import pdb; pdb.set_trace()
    vetor_amostra = shuffle(vetor)[:qtd]
    print(len(vetor_amostra))
    return 0
file_object  = open('wine.data', 'r')
linhas = file_object.readlines()

la = gera_vetor_amostra(linhas,134)
