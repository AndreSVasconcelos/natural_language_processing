import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS

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

# Visualizar categorias
#sns.countplot(treinamento_bd['emocao'], label='Contagem')

# Pre-processamento dos textos
pontuacoes = string.punctuation
pln = spacy.load('pt_core_news_sm')
treinamento_bd['texto'] = treinamento_bd['texto'].apply(preprocessamento)

# Tratamento das classes
treinamento_bd_final = tratamento_classes(treinamento_bd)

# Criar modelo de classificação
modelo = spacy.blank('pt')
categorias = modelo.create_pipe('textcat')
categorias.add_label('ALEGRIA')
categorias.add_label('MEDO')
modelo.add_pipe(categorias)
historico = []
for epoca in range(1000):
    random.shuffle(treinamento_bd_final)
    losses = {}
    for batch in spacy.util.minibatch(treinamento_bd_final, 30):
        textos = [nlp(texto) for texto, entities in batch]
        annotations = [entities for texto, entities in batch]
        modelo.update(textos, annotations, losses=losses)
    
    if epoca % 100 == 0:
        print(losses)
        historico.append(losses)