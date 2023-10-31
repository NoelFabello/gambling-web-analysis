# %%
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# %% [markdown]
# # Scraping
# ## Extracción de datos de la web ludopatía.org
# Se obtienen páginas de conversaciones del foro "General" de la página.  
# De estas conversaciones se almacenan todos los mensajes, con cada usuario, fecha y título de la conversación. Debido a que hay demasiados mensajes, se han limitado a 20 páginas de conversaciones.
# ## Este scraping se divide en 2 partes
# ### Parte 1
# Se obtienen los enlaces de todas las conversaciones que se van a scrapear.

# %%
base_url = 'https://www.ludopatia.org/forum/default.asp'
response = requests.get(base_url)
# Parser HTML
soup = BeautifulSoup(response.content, 'html.parser')
# Se obtiene el apartado del foro General, donde más mensajes hay
forum = soup.find('a', string='General')
general_forum_link = urljoin(base_url, forum.get('href'))
general_forum_html = requests.get(general_forum_link)
soup = BeautifulSoup(general_forum_html.content, 'html.parser')
# Se recorren los links de foro evitando la flecha de siguiente mensaje, tomando todas las conversaciones
forum_types = []
for link in soup.find_all('a'):
    if 'forum' in link.get('href') and link.find('img') is None:
        page_actual = link.get('href')
        page_link = urljoin(base_url, page_actual)
        forum_types.append(page_link)
# Se toma el foro que se desea scrapear
selected_forum = forum_types[0]
forum_content = requests.get(selected_forum)
forumSoup = BeautifulSoup(forum_content.content, 'html.parser')
msg_list = []
next_page = True
# Número de páginas a scrapear
cont = 20
# Se recorren todas las páginas del foro guardando los enlaces a los mensajes
while next_page:
    links = forumSoup.find_all('a')
    links = [link for link in links if 'forum' in link.get('href') and link.find('img') is None]
    
    if 'Siguiente' not in links[len(links)-1].text:
        next_page = False
    
    if cont == 1:
        next_page = False
    else:
        cont = cont -1
    
    for msg in links:
        if 'Anterior' not in msg.text and 'Siguiente' not in msg.text:
            msg_actual = msg.get('href')
            msg_link = urljoin(page_link, msg_actual)
            msg_list.append(msg_link)
            
    next_page_link = urljoin(base_url, links[len(links)-1].get('href'))
    next_page_response = requests.get(next_page_link)
    forumSoup = BeautifulSoup(next_page_response.content, 'html.parser')
    
msg_list = [string for string in msg_list if not string.endswith("TPN=1")]
print('Se han obtenido ', len(msg_list), ' páginas de conversaciones')

# %% [markdown]
# ### Parte 2
# Se obtiene la información y se crea el dataframe en pandas con toda la información.  
# Si la obtención de información de mensajes da error, simplemente se pasa al siguiente. Se ha tomado esta decisión debido a que la cantidad de mensajes es más que suficiente.

# %%
# Creación del dataframe en el que se almacenará la información
df = pd.DataFrame({'user': [], 'date': [], 'title': [], 'text':[]})
# Patrones para encontrar el título de la conversación y la fecha
date_pattern = re.compile('Escrito el:.*[0-9]{2}:[0-9]{2}')
title_pattern = re.compile('Tema:.*')
# Se recorre toda la colección de conversaciones obtenida en la parte anterior
for url in msg_list:
    try:
        msg = requests.get(url)
        forumSoup = BeautifulSoup(msg.content, 'html.parser')
        body = forumSoup.find('body')
        title = title_pattern.search(body.text)[0].split('Tema:')[1].strip()
        forum_messages = body.find_all('table')[5].find_all('table')[0]
        # Obtención de las filas con los mensajes de la conversación
        table_rows = forum_messages.find_all('tr', recursive = False)
        for i in table_rows:
            user_list = i.find_all('span', {'class':'bold'})
            # Toda fila con mensaje, tiene nombre de usuario, por lo que si no se ha encontrado el nombre, no es un mensaje
            if len(user_list) > 0:
                # Obtención de los datos del mensaje
                user = user_list[0].text
                text = i.find_all('td', {'class':'text'})[0].text
                text = text.split('__________________')[0]
                text = text.split('Editado por')[0]
                date = date_pattern.search(text)[0].split(': ')[1]
                text = re.split(date_pattern, text)[1].strip()
                if len(text)>0:
                    new_row = pd.DataFrame({'user': [user], 'date': [date], 'title': [title], 'text': [text]})
                    df = pd.concat([df, new_row], ignore_index=True)
    except Exception as e:
        continue

# %% [markdown]
# ## Observación del dataset resultado
#   
# Como se puede observar, en el dataset resultado se han obtenido 8158 mensajes

# %%
df

# %% [markdown]
# ## Guardado y carga del dataframe en un archivo .json  
# Destacar la importancia de guardar y cargar los datos con la codificación de caracteres correcta.

# %%
df.to_json('/home/noel/Documentos/NoEstructurada/corpus.json', orient='split', force_ascii=False)

# %%
df = pd.read_json('/home/noel/Documentos/NoEstructurada/corpus.json', orient='split', encoding='utf-8')
df

# %% [markdown]
# ## Aplicación del tf/idf  
# El tf/idf es un indicador que permite calcular la importancia de una palabra en una colección de documentos. Utiliza la frecuencia del término en los textos, compensando el valor de cada término con el número de documentos en el que aparece la palabra.  
#    
# Se calcula el tf/idf de las palabras de todo el corpus eliminando las palabras menores de 4 letras, las que aparezcan en menos de 10 documentos y las stopwords.  
# Luego se ordenan las palabras según este valor y se obtienen las 50 primeras.

# %%
stop_words=nltk.corpus.stopwords.words('spanish')
vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=10, token_pattern=r'\b[a-zA-ZÁÉÍÓÚáéíóú]{4,}\b')
tfidf = vectorizer.fit_transform(df['text'])
df_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
top_tfidf_words = df_tfidf.sum().sort_values(ascending=False)[:50]
print(top_tfidf_words)

# %% [markdown]
# ## Term Frecuency(TF)  
# La frecuencia de un término es el número de veces que aparece en el conjunto de documentos.  
#   
# Para calcular el tf de los términos de todos los documentos se utiliza un contador de palabras. Se aplican las mismas restricciones que en el caso del tf/idf y se obtienen las 100 palabras que tienen mayor tf ordenadas de mayor a menor.

# %%
count_vectorizer = CountVectorizer(stop_words=stop_words, min_df=10, token_pattern=r'\b[a-zA-ZÁÉÍÓÚáéíóú]{4,}\b')
counter = count_vectorizer.fit_transform(df['text'])
bow = pd.DataFrame(counter.toarray(), columns=count_vectorizer.get_feature_names_out())
top_frecuency_words = bow.sum().sort_values(ascending=False)[:100]
for i in range(len(top_frecuency_words)):
    print(i, '   ' ,top_frecuency_words.index[i], ' ', top_frecuency_words[i])

# %% [markdown]
# # KeyBERT  
# Se carga el modelo y a través de la función 'extract_keywords' se obtiene conjunto de listas de tuplas (una lista por documento, y una tupla por palabra del documento) con las palabras y sus puntuaciones correspondientes.  
# Esta puntuación representa la importancia de la palabra en el texto. 

# %%
stop_words=nltk.corpus.stopwords.words('spanish')
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
model = KeyBERT(model=model)
keywords = model.extract_keywords(df['text'], stop_words=stop_words)

# %% [markdown]
# ## Acumulación de puntuaciones  
# Para poder ordenar las palabras según su importancia, primero se crea un diccionario donde existe una clave por palabra y en el valor de la clave se acumula la puntuación correspondiente a la palabra.

# %%
distances_keywords = {}
for sentence in keywords:
    for tuple in sentence:
        if tuple[0] in distances_keywords.keys():
            distances_keywords[tuple[0]] += tuple[1]
        else:
            distances_keywords[tuple[0]] = tuple[1]

# %% [markdown]
# Una vez acumuladas las puntuaciones de las palabras, ya se pueden ordenar según su importancia

# %%
sorted_words = sorted(distances_keywords.items(), key= lambda x : x[1], reverse = True)
print(sorted_words)

# %% [markdown]
# # Creación de la nube de palabras
# 
# En este caso, la nube de palabras se crea a partir de los términos y las puntuaciones extraídas del apartado anterior, utilizando el modelo de Keybert.  
#   
# Se utiliza el método 'generate_from_frequencies' para que las palabras tengan tamaño proporcional a la puntuación correspondiente obtenida.

# %%
wordcloud = WordCloud(width=1200, height=1200, background_color='black', min_font_size=10)
wordcloud.generate_from_frequencies(distances_keywords)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


