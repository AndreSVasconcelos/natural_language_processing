# LIBs
import bs4 as bs
import urllib.request
import nltk
import spacy

# Carregamento e limpeza de um texto de uma url
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Harmonia_(m%C3%BAsica)')
dados = dados.read()
dados_html = bs.BeautifulSoup(dados, 'lxml')
paragrafos = dados_html.find_all('p')
# Remove tags html restantes e concatena texto em uma única string em letras minusculas
conteudo = ''
for p in paragrafos:
    conteudo += p.text
conteudo = conteudo.lower()


