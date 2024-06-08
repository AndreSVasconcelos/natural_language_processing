import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.training import Example
from sklearn.metrics import confusion_matrix, accuracy_score

# Funções
def preprocessamento(texto):
    documento = pln(texto.lower())
    lista = []
    for token in documento:
        lista.append(token.lemma_)
    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in string.punctuation]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

def tratamento_classes(bd):
    return_bd = []
    for texto, emocao in zip(bd['texto'], bd['emocao']):
        if emocao == 'alegria':
            dic = ({'ALEGRIA': True, 'MEDO': False})
        elif emocao == 'medo':
            dic = ({'ALEGRIA': False, 'MEDO': True})
        return_bd.append([texto, dic.copy()])
    return return_bd

# Carregar base de dados
treinamento_bd = pd.read_csv('./dados/base_treinamento.txt', encoding='utf-8')
teste_bd = pd.read_csv('./dados/base_teste.txt', encoding='utf-8')

# Pre-processamento dos textos
pontuacoes = string.punctuation
pln = spacy.load('pt_core_news_sm')
treinamento_bd['texto'] = treinamento_bd['texto'].apply(preprocessamento)

# Tratamento das classes
treinamento_bd_final = tratamento_classes(treinamento_bd)

# Criar modelo de classificação
modelo = spacy.blank('pt')
textcat = modelo.add_pipe("textcat")
textcat.add_label("ALEGRIA")
textcat.add_label("MEDO")
historico = []
modelo.begin_training()
for epoca in range(200):
  random.shuffle(treinamento_bd_final)
  losses = {}
  for batch in spacy.util.minibatch(treinamento_bd_final, 30):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
            textos, annotations
        )]
    modelo.update(examples, losses=losses)
  if epoca % 100 == 0:
    print(losses)
    historico.append(losses)

historico_loss = []
for i in historico:
  historico_loss.append(i.get('textcat'))

historico_loss = np.array(historico_loss)

plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

# Salvar o modelo
modelo.to_disk("modelo")

# Realizando testes
modelo_carregado = spacy.load("modelo")
texto_positivo_exemplo = 'eu adoro cor dos seus olhos'
texto_negativo_exemplo = 'estou triste'
texto_positivo_exemplo = preprocessamento(texto_positivo_exemplo)
texto_negativo_exemplo = preprocessamento(texto_negativo_exemplo)
previsao_pos = modelo_carregado(texto_positivo_exemplo)
previsao_neg = modelo_carregado(texto_negativo_exemplo)
print('============================================================')
print(f'Frase: {texto_positivo_exemplo} | Previsão: {previsao_pos.cats}') # print(previsao_pos.cats)
print(f'Frase: {texto_negativo_exemplo} | Previsão: {previsao_neg.cats}') # print(previsao_neg.cats)

# Avaliando modelo na base de treinamento
previsoes = []
for texto in treinamento_bd['texto']:
  #print(texto)
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
    previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')

previsoes_final = np.array(previsoes_final)
respostas_reais = treinamento_bd['emocao'].values
print('============================================================')
print(f'Respostas reais: {respostas_reais}')
accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
print('============================================================')
print('Confusion matrix da base de treinamento:')
print(cm)

# Avaliando modelo na base de teste
teste_bd['texto'] = teste_bd['texto'].apply(preprocessamento) # Pre-processamento dos textos
previsoes = []
for texto in teste_bd['texto']:
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
    previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')

previsoes_final = np.array(previsoes_final)
respostas_reais = teste_bd['emocao'].values
accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
print('============================================================')
print('Confusion matrix da base de teste:')
print(cm)