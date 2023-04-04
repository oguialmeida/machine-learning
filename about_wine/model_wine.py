import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#importar os dados
#from google.colab import files
#uploaded = files.upload()
#arquivo = pd.read_csv(io.BytesIO(uploaded['wine_dataset_henri.csv']))

arquivo = pd.read_csv('wine_dataset_henri.csv')
arquivo.head()

#convertendo a coluna de resultados para binários
#red - 0
#white - 1
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)
arquivo.head()

x = arquivo[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']]
x.head()

y = arquivo[['style']]
y.head()

treino_x, teste_x, treino_y, teste_y = train_test_split(x,y)

modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

print (accuracy_score(teste_y, previsoes) * 100)


from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

print (accuracy_score(teste_y, previsoes) * 100)
