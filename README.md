# projeto-Python-DS
Projeto de Data Science via Python

# Projeto de Preços para imóveis na plataforma Airbnb

## Contexto

No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.

Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.

Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)

Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.

## Objetivo

Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.

Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.

## Inspiração

As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro

Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados. A base está na pasta 'dataset'

### Expectativas Iniciais

- A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
- Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro

Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

### Bibliotecas utilizadas:

import pandas as pd
import pathlib
import numpy as np # para calculos
import seaborn as sns # para gráficos
import matplotlib.pyplot as plt # para gráficos
import plotly.express as px # mapa de calor
from sklearn.metrics import r2_score, mean_squared_error # para calculos estatísticos dos modelos
from sklearn.linear_model import LinearRegression # modelo de regressao linear
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor # modelo de randomforest e extratrees
from sklearn.model_selection import train_test_split # para segregar a base entre treino e teste

Agora vamos importar as bases de dados. Nosso objetivo é anexar todos os arquivos em uma única base, adicionando as colunas de mes e ano

Para isso, vamos criar um dicionário com os 3 primeiros caracteres com os nomes dos meses e o número correspondente daquele mês, pois assim conseguimos identificar de que mês é o arquivo .csv que está na pasta

Para cada arquivo da base de dados a gente vai importar o arquivo e criar uma coluna na base de dados com o mês e uma coluna com o ano de cada informação


meses = {'jan': 1, 'fev':2, 'mar':3, 'abr': 4, 'mai':5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12} # dicionario com as 3 primeiras letras do mes

caminho_bases = pathlib.Path('dataset') # caminho dos arquivos em uma variável

base_airbnb = pd.DataFrame() # criamos um dataframe vazio para que receba todas as bases

# Queremos adicionar uma coluna com o numero do mes e o numero do ano em cada dataframe. Para isso, precisamos pegar o mes e o ano do nome de cada arquivo:
for arquivo in caminho_bases.iterdir(): # para cada arquivo na pasta
    nome_mes = arquivo.name[:3] # o nome do mes são os 3 primeiros dígitos do nome do arquivo
    mes = meses[nome_mes] # a variavel mes recebe o valor do dicionario – seu numero - meses de acordo com o nome do mes
    
    ano = arquivo.name[-8:] # o ano sao os ultimos 8 digitos
    ano = int(ano.replace('.csv', '')) # retira a extensão .csv e transforma o texto em um numero
    
    df = pd.read_csv(caminho_bases / arquivo.name, low_memory=False) # cria o df para cada arquivo csv da pasta
    df['ano'] = ano # cria a coluna de ano
    df['mes'] = mes # cria a coluna de mes

    base_airbnb = base_airbnb.append(df) # adiciona o novo df para a nova base

display(base_airbnb)

### Agora vamos começar os tratamentos

- Como temos muitas colunas, nosso modelo pode acabar ficando muito lento.
- Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão, por isso, vamos excluir algumas colunas da nossa base
- Tipos de colunas que vamos excluir:
    1. IDs, Links e informações não relevantes para o modelo
    2. Colunas repetidas ou extremamente parecidas com outra (que dão a mesma informação para o modelo. Ex: Data x Ano/Mês
    3. Colunas preenchidas com texto livre -> Não rodaremos nenhuma análise de palavras ou algo do tipo
    4. Colunas em que todos ou quase todos os valores são iguais
    
- Para isso, vamos criar um arquivo em excel com os 1.000 primeiros registros e fazer uma análise qualitativa, olhando as colunas e identificando quais são desnecessárias


print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')

# Analisamos via Excel as colunas que vamos retirar, mas colocamos algumas observações

#Obs: para pesquisar o que uma coluna tem, precisamos fazer usar o metodo value_counts()
print(base_airbnb['experiences_offered'].value_counts())
print('----------------')
#Obs2: para saber se duas colunas tem valores iguais:
print((base_airbnb['host_listings_count']==base_airbnb['host_total_listings_count']).value_counts())
print('----------------')
#Obs3: para saber quantas celulas vazias tem:
print(base_airbnb['square_feet'].isnull().sum())
print('----------------')

# No arquivo, retiramos as seguintes colunas: Com o arquivo, retiramos as seguintes colunas:
# [id, listing_url, scrape_id, name, summary, space, description, neighborhood_overview, notes, transit, interaction, house, thumbnail, medium, picture, xl_picture, host_id, host_link, host_name, host_since, host_location, host_about, host_response_time, host_acceptance_rate, host_thumbnail, host_picture, host_neighborhood, host_listing_count, host_verification, host_profile_picture, host_identity_verify, street, neighborhood, neighborhood, neighborhood, city, state, zipcode, market, smart, country, country, is_location, square_feet, weekly, monthly, calendar_updated, has_availability, availability_30, availability_60, availability_90, availability_365, calendar_last_scraped, first_review, last_review, requires_license, license, jurisdiction, require, require, calculated_host_listing_count, reviews_por_month, minimum, max, minimun, max, minimum, max, number, calculation, calculation, calculation, host_listings_count_shared]


### Depois da análise qualitativa das colunas, levando em conta os critérios explicados acima, ficamos com as seguintes colunas:

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas] # filtrar a base apenas com as colunas desejadas
print(list(base_airbnb.columns))
display(base_airbnb)

### Tratar Valores Faltando

- Visualizando os dados, percebemos que existe uma grande quantidade em dados faltantes. As colunas com mais de 300.000 valores NaN serão excluídas da análise
- Para as outras colunas, como temos muitos dados (mais de 900.000 linhas), vamos excluir as linhas que contém dados NaN

# exclusão de colunas:

for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())

# exclusão de linhas:

base_airbnb = base_airbnb.dropna()

print(base_airbnb.shape)
print(base_airbnb.isnull().sum())

### Verificar Tipos de Dados em cada coluna

- Precisamos fazer isso porque às vezes o Python está lendo como texto alguma coluna que deveria ser um número, então precisamos corrigir

print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])

- Como preço e extra people estão sendo reconhecidos como objeto (ao invés de ser um float) temos que mudar o tipo de variável da coluna e retirar os caracteres das células

#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)
#verificando os tipos
print(base_airbnb.dtypes)

### Análise Exploratória e Tratamento de Outliers

- Vamos basicamente olhar feature por feature (coluna a coluna) para:
    1. Ver a correlação entre as features e decidir se manteremos todas as features que temos. Uma correlação muito forte pode indicar uma coluna a excluir.
    2. Excluir outliers (usaremos como regra: valores abaixo de Q1 - 1.5 x Amplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
    3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vão nos ajudar e se devemos excluir
    
- Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.

- Depois vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)

- Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.


Exemplo de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listings_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do modelo.

plt.figure(figsize=(20, 15))
sns.heatmap(base_airbnb.corr(), annot=True, annot_kws={"size": 10}, cmap='Greens', fmt='.2f', linewidths=.5, square=True, cbar_kws={"shrink": .75})
matriz_correlacao = base_airbnb.corr()
matriz_correlacao_formatada = matriz_correlacao.round(2)
print(matriz_correlacao_formatada)

Analisando as correlações, vamos manter todas as colunas.

### Definição de Funções para Análise de Outliers

Vamos definir algumas funções para ajudar na análise de outliers das colunas. Faremos uma função para usar em cada uma das colunas. Essa função me trará o primeiro quartil, o terceiro quartil, a amplitude e os limites inferior e superior de cada coluna. Valores abaixo do limite inferior e acima do limite superior serão considerados outliers.

def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df,  linhas_removidas

Funções para gráficos:

def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)

def grafico_barra(coluna):  
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))

##### Vamos analisar a coluna de preço.

diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])

Como estamos construindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas de apartamentos de altíssimo luxo, que não é o nosso objetivo principal. Por isso, podemos excluir esses outliers.

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(f'{linhas_removidas} linhas removidas')

diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])
print(base_airbnb.shape)

#### extra_people
Valor a pagar para pessoa adicional

diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])

diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{linhas_removidas} linhas removidas')

histograma(base_airbnb['extra_people'])
print(base_airbnb.shape)

Agora vamos analisar as informações discretas. As colunas são: host_listing_count, accomodates, bathrooms, bedrooms, beds, guests_included, minimum_nights, maximum_nights, number_of_reviews.

### host_listing_count
Quantos apartamentos o host tem no airbnb

diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])

Analisando a coluna host_listing_count, podemos excluir os outliers porque para o objetivo do nosso projeto porque hosts com mais de 6 imóveis no airbnb não é o público alvo do objetivo do projeto (imagino que sejam imobiliários ou profissionais que gerenciam imóveis no airbnb)


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas')

### accommodates
Quantas pessoas o apartamento acomoda

diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])

- Pelo mesmo motivo do "host_listings_count" vamos excluir os outliers dessa coluna porque apartamentos que acomodam mais de 9 pessoas não são o nosso foco, nosso objetivo aqui é para imóveis comuns.

- Caso a gente quisesse incluir apartamentos de alto padrão a gente poderia manter essa variável ou então construir um modelo só focado em imóvei com mais de 9 hóspedes

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{linhas_removidas} linhas removidas')

### bathrooms

diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())

- Pelo mesmo motivo dos anteriores, vamos excluir os outliers nos banheiros

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{linhas_removidas} linhas removidas')

### bedrooms

diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])

Como são poucos os aptos com mais de 3 quartos, vamos retirar do modelo

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{linhas_removidas} linhas removidas')

### beds

diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])

Como são poucos os aptos com mais de 6 camas, vamos retirar do modelo

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(f'{linhas_removidas} linhas removidas')

### guests_included
Quantas pessoas já estão incluídas no pacote

diagrama_caixa(base_airbnb['guests_included'])
grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())

Os limites foram 1 e 1. A grande maioria coloca esse padrão como 1. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço, por isso, me parece melhor excluir a coluna da análise:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape

### minimum_nights

diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])

- Estamos querendo um modelo que ajude a precificar apartamentos comuns como uma pessoa comum gostaria de disponibilizar. No caso, apartamentos com mais de 8 noites como o "mínimo de noites" podem ser apartamentos de temporada ou ainda apartamentos para morar, em que o host exige pelo menos 1 mês no apartamento.

- Por isso, vamos excluir os outliers dessa coluna

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas')

### maximum_nights

diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])

- Essa coluna não parece que vai ajudar na análise. Isso porque parece que quase todos os hosts não preenchem esse campo de maximum nights, então ele não parece que vai ser um fator relevante.

- É melhor excluirmos essa coluna da análise

base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape

### number_of_reviews           

diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])

- Aqui poderíamos tomar algumas decisões diferentes. Eu vou tomar uma decisão de tirar essa característica da análise, por alguns motivos:

    1. Se excluirmos os outliers, vamos excluir as pessoas que tem a maior quantidade de reviews (o que normalmente são os hosts que têm mais aluguel). Isso pode impactar muito negativamente o nosso modelo
    2. Pensando no nosso objetivo, se eu tenho um imóvel parado e quero colocar meu imóvel lá, é claro que eu não tenho review nenhuma. Então talvez tirar essa característica da análise pode na verdade acabar ajudando.

base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape

### Tratamento de Colunas de Valores de Texto

### - property_type 
Casa, apartamento, hotel...

print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

- Aqui a nossa ação não é "excluir outliers", mas sim agrupar valores que são muito pequenos.

- Todos os tipos de propriedade que têm menos de 2.000 propriedades na base de dados, eu vou agrupar em um grupo chamado "outros". Acho que isso vai facilitar o nosso modelo

tabela_tipos_casa = base_airbnb['property_type'].value_counts() # fazemos uma tabela com a contagem de quantos itens cada categoria tem
colunas_agrupar = [] # lista vazia que agrupará os valores

for tipo in tabela_tipos_casa.index: # para cada tipo de casa que vemos na tabela Tabela_tipos_casa
    if tabela_tipos_casa[tipo] < 2000: # se o valor na tabela for menor que 2000,
        colunas_agrupar.append(tipo) # coloca na lista colunas_agrupar
print(colunas_agrupar)

for tipo in colunas_agrupar: # para cada categoria na lista colunas_agrupar, 
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros' # nas linhas em que a categoria aparece na coluna property_type, colocamos o valor ´Outros´

print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

### - room_type 

print(base_airbnb['room_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

Não será necessário mudar. Já existem poucas categorias.

### - bed_type 

print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

Vamos agrupar as outras 4 colunas em uma chamada Outros, pois a coluna Real Bed é muito maior que o resto

# agrupando categorias de bed_type
tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

### - cancellation_policy 

print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

Podemos agrupar 3 categorias de ´strict´ em apenas uma.

# agrupando categorias de cancellation_pollicy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x='cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

### - amenities 
Utensílios e ferramentas que o host disponibiliza no apartamento

A coluna amenities possui diversos valores em cada célula. Além disso, não há um padrão nesses valores.
Podemos analisar de forma que, quanto mais amenities um apartamento possui, mais caro ele pode ser. Portanto, podemos contar quantos amenities cada apto tem.
Depois, podemos adicionar uma coluna ´numero_de_amenities´ e retirar a coluna de ´amenities´:


#análise de primeira célula da coluna amenities

print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(','))) 

# colocar uma coluna de numero de amenities
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)

# retirar a coluna de amenities
base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape

### n_amenities

diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])

Então, essa virou uma coluna de valor numérico e, como todas as outras colunas de valores numéricos, eu excluirei outliers com os mesmos modelos anteriores

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{linhas_removidas} linhas removidas')

### Visualização de Mapa das Propriedades

Vamos criar um mapa que exibe um pedaço da nossa base de dados aleatório (50.000 propriedades) para ver como as propriedades estão distribuídas pela cidade e também identificar os locais de maior preço 

amostra = base_airbnb.sample(n=50000) # pegamos apenas uma amostra da base, para que fique mais rápido.

centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()} # quando o mapa abrir, podemos escolher qual será o centro do mapa. 
#OBS: A sintaxe amostra.latitude.mean é a mesma coisa que amostra[´latitude´].mean
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()

### Encoding

Precisamor Ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true e false, etc.)

- Features de Valores True ou False, vamos substituir True por 1 e False por 0.
- Features de Categoria (features em que os valores da coluna são textos) vamos utilizar o método de encoding de variáveis dummies

colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready'] # colocamos as colunas t/f em uma lista
base_airbnb_cod = base_airbnb.copy() # copiamos a base para podermos modifica-la
for coluna in colunas_tf: # para cada coluna de T/F
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1 # Na base modificada, onde as linhas da coluna forem ‘t’, a coluna toda passa a ser 1 
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0 # Na base modificada, onde as linhas da coluna forem ‘f’, a coluna toda passa a ser 0

colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias) # O pandas faz isso para a gente, via método get_dummies, precisando passar apenas o dataframe e as colunas.
display(base_airbnb_cod.head())

### Modelo de Previsão

- Primeiramente, precisamos avaliar se é um problema de classificação ou regressão. Dado que queremos encontrar um valor numérico, faremos regressão.

- Métricas de Avaliação

Vamos usar aqui o R² que vai nos dizer o quão bem o nosso modelo consegue explicar o preço. Isso seria um ótimo parâmetro para ver o quão bom é nosso modelo <br>
-> Quanto mais próximo de 100%, melhor

Vou calcular também o Erro Quadrático Médio, que vai mostrar para gente o quanto o nosso modelo está errando. <br>
-> Quanto menor for o erro, melhor

def avaliar_modelo(nome_modelo, y_teste, previsao): # nome_modelo = nome, y_teste = resposta do teste, 
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'

- Escolha dos Modelos a Serem Testados
    1. RandomForest
    2. LinearRegression
    3. Extra Tree
    
Esses são alguns dos modelos que existem para fazer previsão de valores numéricos (o que chamamos de regressão). Estamos querendo calcular o preço, portanto, queremos prever um valor numérico.

Assim, escolhemos esses 3 modelos.

# Criar variáveis e dicionário de modelos e escolher as variáveis x e y:

modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price'] # o preço será a variável y
X = base_airbnb_cod.drop('price', axis=1) # o x serão todas as variáveis, menos o preço

- Separa os dados em treino e teste + Treino do Modelo

Separamos em 80/20 para treino e teste.
O treino serve para fazer a máquina aprender.
O teste serve para ver se a máquina aprendeu mesmo.

Não colocamos tudo no treino porque o modelo pode ficar ruim para novos dados (overfitting – bom demais somente para os dados da base atual)

# Nomear as 4 variáveis que receberão os outputs do módulo de segregação:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10) # o random_state é um número aleatório que 
# serve como base para o código segregar as bases. Sem ele, todas as vezes que rodássemos o código as bases seriam 
# diferentes. Colocando uma seed, a segregação seguirá um ´padrão´.


# Usaremos um FOR para treinar e testar cada modelo.
# Como o método .items() traz uma tupla, precisamos fazer um unpack no FOR.

for nome_modelo, modelo in modelos.items():
    #treinar o modelo
    modelo.fit(X_train, y_train)
    #testar o modelo
    previsao = modelo.predict(X_test) # x_test é a base  x usada para teste
    print(avaliar_modelo(nome_modelo, y_test, previsao)) # o y_test é a resposta do teste do modelo

- Modelo Escolhido como Melhor Modelo: ExtraTressRegressor

    Esse foi o modelo com maior valor de R² e ao mesmo tempo o menor valor de RSME. Como não tivemos uma grande diferença de velocidade de treino e de previsão desse modelo com o modelo de RandomForest (que teve resultados próximos de R² e RSME), vamos escolher o Modelo ExtraTrees.
    
    O modelo de regressão linear não obteve um resultado satisfatório, com valores de R² e RSME muito piores do que os outros 2 modelos.
    
- Resultados das Métricas de Avaliaçõ no Modelo Vencedor:<br>
Modelo ExtraTrees:<br>
R²:97.52%<br>
RSME:41.78

### Entendimento e Ajustes ao Modelo

print(modelo_et.feature_importances_)
print(X_train.columns)

importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns) # tabela com as duas colunas
importancia_features = importancia_features.sort_values(by=0, ascending=False) # organizando a coluna
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90) # colocar o rotulo em 90 graus

Análise:

Localização conta de forma considerável. <br>
Número de amenities é um fator que pode aumentar o preço <br>
´Is_business_travel_ready´ = não teve impacto no modelo. Vamos retirar

base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1) # retiramos a coluna

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao)) # treinamos e testamos o modelo extratrees novamente

Vamos testar retirar as colunas de bed type:

base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))

print(previsao)

# Deploy do Projeto

- Passo 1 -> Criar arquivo do Modelo (joblib)<br>
- Passo 2 -> Escolher a forma de deploy:
    - Deploy apenas para uso direto Streamlit
- Passo 3 -> Outro arquivo Python
- Passo 4 -> Importar streamlit e criar código
- Passo 5 -> Atribuir ao botão o carregamento do modelo
- Passo 6 -> Deploy feito

# Criar um arquivo csv para os dados e um arquivo para o modelo:

X['price'] = y
X.to_csv('dados.csv')

import joblib
joblib.dump(modelo_et, 'modelo.joblib') # cria o arquivo do modelo formato joblib
