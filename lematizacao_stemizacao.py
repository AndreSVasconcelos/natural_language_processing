# LIBs
import bs4 as bs
import urllib.request
import nltk
import spacy

# Carregando Spacy em Português -> Lematização
pln = spacy.load('pt_core_news_sm')

# Criar documento de texto a ser analisado
documento = pln('Batatinha quando nasce se esparrama pelo chão')

# Stemização trabalha com radicais
stemmer = nltk.stem.RSLPStemmer()

# Verifica como o spacy classificou as palavras
for token in documento:
    print(token.text, token.lemma_, token.pos, stemmer.stem(token.text))

