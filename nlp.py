# LIBs
import bs4 as bs
import urllib.request
import nltk
import spacy

# Carregando Spacy em Português
pln = spacy.load('pt_core_news_sm')

# Criar documento de texto a ser analisado
documento = pln('Batatinha quando nasce se esparrama pelo chão')

# Verifica como o spacy classificou as palavras
for token in documento:
    print(token.text, token.pos)

