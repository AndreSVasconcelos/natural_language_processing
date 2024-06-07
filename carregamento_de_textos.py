# LIBs
import bs4 as bs
import urllib.request
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from IPython.core.display import HTML
from spacy import displacy

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

# Busca de palavras com spacy
pln = spacy.load('pt_core_news_sm')
string='acorde'
token_pesquisa = pln(string)
matcher = PhraseMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)
doc = pln(conteudo)
matches = matcher(doc)

# Apresentando resultado
numero_palavras = 10
#display(HTML(f'<h1>{string.upper()}</h1>'))
#display(HTML(f"""<p><strong>Resultados encontrados:</strong> {len(matches)}</p>"""))

texto = ''
for i in matches:
    inicio = i[1] - numero_palavras
    if inicio < 0:
        inicio = 0
    texto += str(f"...{doc[inicio:i[2] + numero_palavras]}...").replace(string, string.upper())
    texto += '\n\n'
print(texto)

# Extração de entidades nomeadas
for entidades in doc.ents:
    print(entidades.text, entidades.label_)

displacy.render(doc, style='ent')