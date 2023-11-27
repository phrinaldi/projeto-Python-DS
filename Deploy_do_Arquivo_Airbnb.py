#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st
import joblib

x_numericos = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 'extra_people': 0,
               'minimum_nights': 0, 'ano': 0, 'mes': 0, 'n_amenities': 0, 'host_listings_count': 0}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_listas = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Outros', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'cancellation_policy': ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']
            }

dicionario = {} # um dicionario que tenha todos os valores da lista, exemplo: property_type_apartment. Isso serve para colocarmos um valor de 1 caso seja selecionado, dado que é uma variável dummy
for item in x_listas: # uma das 3 chaves do dicionario
    for valor in x_listas[item]: # para cada valor na lista dentro da chave do dicionario,
        dicionario[f'{item}_{valor}'] = 0 # adicionar ao dicionario auxiliar uma chave “chave_valor_da_lista”


for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor

for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor == "Sim":
        x_tf[item] = 1
    else:
        x_tf[item] = 0
    
    
for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dicionario[f'{item}_{valor}'] = 1
    

botao = st.button('Prever Valor do Imóvel')

if botao:
    dicionario.update(x_numericos)
    dicionario.update(x_tf)
    valores_x = pd.DataFrame(dicionario, index=[0]) # precisamos de um dataframe pq nosso modelo é um dataframe. Colocamos o index pq senão o código não roda
    
    dados = pd.read_csv("dados.csv")
    colunas = list(dados.columns)[1:-1] # retiramos as colunas de índice e de preço
    valores_x = valores_x[colunas] # alteramos a ordem das colunas para ficar de acordo com o arquivo dados.csv
    
    modelo = joblib.load('modelo.joblib')
    preco = modelo.predict(valores_x)
    st.write(preco[0])
    

