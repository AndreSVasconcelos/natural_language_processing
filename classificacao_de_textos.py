import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np

# Carregar base de dados
treinamento_bd = pd.read_csv('./dados/base_treinamento.txt', encoding='utf-8')

# Visualizar categorias
sns.countplot(treinamento_bd['emocao'], label='Contagem')
