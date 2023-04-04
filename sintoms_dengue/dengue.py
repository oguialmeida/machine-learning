#importar os dados
import io
import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#uploaded = files.upload()

arquivo = pd.read_csv(io.BytesIO(uploaded['dados_dengue.csv']))
arquivo.head()

#convertendo a coluna de resultados para binários
arquivo['sexo'] = arquivo['sexo'].replace('M', 0)
arquivo['sexo'] = arquivo['sexo'].replace('F', 1)

arquivo['manchas_na_pele'] = arquivo['manchas_na_pele'].replace('Nao', 0)
arquivo['manchas_na_pele'] = arquivo['manchas_na_pele'].replace('Sim', 1)

arquivo['vômitos'] = arquivo['vômitos'].replace('Nao', 0)
arquivo['vômitos'] = arquivo['vômitos'].replace('Sim', 1)

arquivo['dor_no_corpo'] = arquivo['dor_no_corpo'].replace('Nao', 0)
arquivo['dor_no_corpo'] = arquivo['dor_no_corpo'].replace('Sim', 1)

arquivo['diarreia'] = arquivo['diarreia'].replace('Nao', 0)
arquivo['diarreia'] = arquivo['diarreia'].replace('Sim', 1)

arquivo['tem_dengue?'] = arquivo['tem_dengue?'].replace('Nao', 0)
arquivo['tem_dengue?'] = arquivo['tem_dengue?'].replace('Sim', 1) 

arquivo.head()

x = arquivo[["idade",	"sexo",	"temperatura",	"manchas_na_pele",	"vômitos",	"dor_no_corpo",	"diarreia",	"tem_dengue?"]]
x.head()

y = arquivo[['tem_dengue?']]
y.head()

# O argumento test_size define a proporção do conjunto de dados a ser usado como conjunto de teste (30%)
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,test_size=0.3, random_state=42)

modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

print(f"A acurácia após a divisão do dataset é de: {format(round(accuracy_score(teste_y, previsoes) * 100, 2))}%")
