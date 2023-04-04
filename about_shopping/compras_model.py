import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#importando os dados de um arquivo externo
dados = pd.read_csv("bd_4.csv")
dados.head()

x = dados[["home", " como_funciona", " contato"]]
x.head()

y = dados[[" comprou;;;;"]]
y.head()

#estanciar o modelo
modelo = LinearSVC()
#treinar o modelo
modelo.fit(x,y)
previsoes = modelo.predict(x)
print("A acurácia sem a divisão do dataset é de: {}%".format(round(accuracy_score(y, previsoes) * 100, 2)))

modelo = LinearSVC()
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y)
# 75% para treinamento
# 25% para teste

# treinando o algoritmo utilizando os dados dividos
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)
print("A acurácia após a divisão do dataset é de: {}%".format(round(accuracy_score(teste_y, previsoes) * 100, 2)))