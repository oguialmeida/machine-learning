from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1 - pelo longo / 0 - pelo curto
#1 - perna curta / 0 - perna longa
#1 - late / 0 - não late
porco1 = [0, 1, 0] # não pelo longo. perna curta e não late
porco2 = [0, 1, 1] # não pelo longo, perna curta, late
porco3 = [1, 1, 0] # pelo longo, perna curta, não late
cachorro1 = [0, 1, 1] # não pelo lono, perna curta, late
cachorro2 = [1, 0, 1] # pelo longo, não perna curta, late
cachorro3 = [1, 1, 1] # pelo longo, perna curta, late
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
# 0 - cachorro
# 1 - porco
treino_y = [1, 1, 1, 0, 0, 0]

modelo = LinearSVC()
#treinar o modelo
modelo.fit(treino_x, treino_y)
animal_misterioso = [0, 0, 1]
result = modelo.predict([animal_misterioso])

if result == 0:
  print ("O animal é um cachorro")
else:
  print ("O animal é um porco")
#no caso anterior, analisamos apenas um animal, mas podemos analisar um conjunto de animais com ML
animal_misterioso_1 = [1,1,1]
animal_misterioso_2 = [1,1,0]
animal_misterioso_3 = [0,1,1]
teste_x = [animal_misterioso_1, animal_misterioso_2, animal_misterioso_3]
teste_y = [0, 1, 1]
previsoes = modelo.predict(teste_x)

resultado = (accuracy_score(teste_y, previsoes))*100
print ("A acurácia do sistema é de: {}%".format(resultado))

#mas como saber se meu algoritmo tem uma acurácia boa?
from sklearn.dummy import DummyClassifier

#uniform, most_frequent
dummy_clf = DummyClassifier(strategy="uniform")

dummy_clf.fit(treino_x, treino_y)

dummy_clf.predict(teste_x)

score = dummy_clf.score(teste_x, teste_y)
score

