import wikipediaapi
import json 
from pathlib import Path
import deep_translator
from deep_translator import GoogleTranslator

file_species_path = "species-categories.json"
with open(file_species_path, 'r', encoding='utf-8') as file:
    especies = json.load(file)

#Cria requisição
wiki_wiki_pt = wikipediaapi.Wikipedia(user_agent = 'Plantas-IA (https://github.com/bordalofelipe/plantas-ia)', language = 'pt')
wiki_wiki_en = wikipediaapi.Wikipedia(user_agent = 'Plantas-IA (https://github.com/bordalofelipe/plantas-ia)', language = 'en')

dicionario_especies_wiki = {}
lista_especies_sem_wiki_pt = []
lista_especies_sem_wiki = []

#Verifica existência da página em português e salva no dicionário
for especie in especies:
    pagina_especie = wiki_wiki_pt.page(especie)
    if pagina_especie.exists() == True:
        dicionario_especies_wiki[especie] = pagina_especie.summary
        #print(f'Adicionou em PT: {especie}')
    else:
        lista_especies_sem_wiki_pt.append(especie)

#Verifica existência da página em inglês, traduz e salva no dicionário
for especie_en in lista_especies_sem_wiki_pt:
    pagina_especie_en = wiki_wiki_en.page(especie_en)
    if pagina_especie_en.exists() == True:
        translated = GoogleTranslator(source='auto', target='pt').translate(pagina_especie_en.summary)
        #print(translated)
        dicionario_especies_wiki[especie_en] = translated
    else:
        lista_especies_sem_wiki.append(especie_en)

dicionario_especies_wiki["especies_sem_wiki"] = lista_especies_sem_wiki

with open("descricoes_especies_wiki.json", "w") as file:
    print("Vai salvar o arquivo")
    json.dump(dicionario_especies_wiki, file)