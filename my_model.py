# Faça um algoritmo de aprendizado de máquina com pelo menos 100 entradas para o treino e pelo menos 
# 15 entradas para validação. Apresente o fator de decisão do algoritmo, isto é, o que ele decide entre 
# um ou outro. Por fim, implemente um classificador dummy e compare a acurácia entre o algoritmo de aprendizado 
# de máquina e a acurácia do dummy.

# Treinamento de IA que identifica qualidade da carne de um frigorifico
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# Critérios I -> Ph Baixo 1 - Ph Alto 0
# Critério II -> Macia 1 - Não macia 0
# Critério II -> Fresca 1 - Não fresca 0
# Critério IV -> Mole 1 - Dura 0

# Carne Ruim - Ph Alto, Não macia, Não fresca, Dura (-2)
# Carne Boa - Ph Baixo, Macia, Fresca, Mole (-1)


# Base de dados
carne_1 = [0, 0, 0, 0]
carne_2 = [0, 0, 1, 0]
carne_3 = [1, 1, 0, 0]
carne_4 = [0, 1, 0, 1]
carne_5 = [1, 1, 0, 0]
carne_6 = [0, 1, 1, 1]
carne_7 = [1, 0, 1, 1]
carne_8 = [1, 1, 0, 1]
carne_9 = [1, 1, 1, 0]
carne_10 = [1, 1, 1, 1]
carne_11 = [0, 1, 1, 1]
carne_12 = [1, 0, 1, 1]
carne_13 = [1, 1, 0, 1]
carne_14 = [1, 1, 1, 0]
carne_15 = [1, 1, 1, 1]
carne_16 = [0, 0, 0, 0]
carne_17 = [0, 0, 1, 0]
carne_18 = [1, 1, 0, 0]
carne_19 = [0, 1, 0, 1]
carne_20 = [1, 1, 0, 0]
carne_21 = [0, 0, 0, 0]
carne_22 = [0, 0, 1, 0]
carne_23 = [1, 1, 0, 0]
carne_24 = [0, 1, 0, 1]
carne_25 = [1, 1, 0, 0]
carne_26 = [0, 1, 1, 1]
carne_27 = [1, 0, 1, 1]
carne_28 = [1, 1, 0, 1]
carne_29 = [1, 1, 1, 0]
carne_30 = [1, 1, 1, 1]
carne_31 = [0, 1, 1, 1]
carne_32 = [1, 0, 1, 1]
carne_33 = [1, 1, 0, 1]
carne_34 = [1, 1, 1, 0]
carne_35 = [1, 1, 1, 1]
carne_36 = [0, 0, 0, 0]
carne_37 = [0, 0, 1, 0]
carne_38 = [1, 1, 0, 0]
carne_39 = [0, 1, 0, 1]
carne_40 = [1, 1, 0, 0]
carne_41 = [0, 0, 0, 0]
carne_42 = [0, 0, 1, 0]
carne_43 = [1, 1, 0, 0]
carne_44 = [0, 1, 0, 1]
carne_45 = [1, 1, 0, 0]
carne_46 = [0, 1, 1, 1]
carne_47 = [1, 0, 1, 1]
carne_48 = [1, 1, 0, 1]
carne_49 = [1, 1, 1, 0]
carne_50 = [1, 1, 1, 1]
carne_51 = [0, 1, 1, 1]
carne_52 = [1, 0, 1, 1]
carne_53 = [1, 1, 0, 1]
carne_54 = [1, 1, 1, 0]
carne_55 = [1, 1, 1, 1]
carne_56 = [0, 0, 0, 0]
carne_57 = [0, 0, 1, 0]
carne_58 = [1, 1, 0, 0]
carne_59 = [0, 1, 0, 1]
carne_60 = [1, 1, 0, 0]
carne_61 = [0, 0, 0, 0]
carne_62 = [0, 0, 1, 0]
carne_63 = [1, 1, 0, 0]
carne_64 = [0, 1, 0, 1]
carne_65 = [1, 1, 0, 0]
carne_66 = [0, 1, 1, 1]
carne_67 = [1, 0, 1, 1]
carne_68 = [1, 1, 0, 1]
carne_69 = [1, 1, 1, 0]
carne_70 = [1, 1, 1, 1]
carne_71 = [0, 1, 1, 1]
carne_72 = [1, 0, 1, 1]
carne_73 = [1, 1, 0, 1]
carne_74 = [1, 1, 1, 0]
carne_75 = [1, 1, 1, 1]
carne_76 = [0, 0, 0, 0]
carne_77 = [0, 0, 1, 0]
carne_78 = [1, 1, 0, 0]
carne_79 = [0, 1, 0, 1]
carne_80 = [1, 1, 0, 0]
carne_81 = [0, 0, 0, 0]
carne_82 = [0, 0, 1, 0]
carne_83 = [1, 1, 0, 0]
carne_84 = [0, 1, 0, 1]
carne_85 = [1, 1, 0, 0]
carne_86 = [0, 1, 1, 1]
carne_87 = [1, 0, 1, 1]
carne_88 = [1, 1, 0, 1]
carne_89 = [1, 1, 1, 0]
carne_90 = [1, 1, 1, 1]
carne_91 = [0, 1, 1, 1]
carne_92 = [1, 0, 1, 1]
carne_93 = [1, 1, 0, 1]
carne_94 = [1, 1, 1, 0]
carne_95 = [1, 1, 1, 1]
carne_96 = [0, 0, 0, 0]
carne_97 = [0, 0, 1, 0]
carne_98 = [1, 1, 0, 0]
carne_99 = [0, 1, 0, 1]
carne_100 = [1, 1, 0, 0]

# Avaliação das carnes
carnes = [
    carne_1,
    carne_2,
    carne_3,
    carne_4,
    carne_5,
    carne_6,
    carne_7,
    carne_8,
    carne_9,
    carne_10,
    carne_11,
    carne_12,
    carne_13,
    carne_14,
    carne_15,
    carne_16,
    carne_17,
    carne_18,
    carne_19,
    carne_20,
    carne_21,
    carne_22,
    carne_23,
    carne_24,
    carne_25,
    carne_26,
    carne_27,
    carne_28,
    carne_29,
    carne_30,
    carne_31,
    carne_32,
    carne_33,
    carne_34,
    carne_35,
    carne_36,
    carne_37,
    carne_38,
    carne_39,
    carne_40,
    carne_41,
    carne_42,
    carne_43,
    carne_44,
    carne_45,
    carne_46,
    carne_47,
    carne_48,
    carne_49,
    carne_50,
    carne_51,
    carne_52,
    carne_53,
    carne_54,
    carne_55,
    carne_56,
    carne_57,
    carne_58,
    carne_59,
    carne_60,
    carne_61,
    carne_62,
    carne_63,
    carne_64,
    carne_65,
    carne_66,
    carne_67,
    carne_68,
    carne_69,
    carne_70,
    carne_71,
    carne_72,
    carne_73,
    carne_74,
    carne_75, 
    carne_76, 
    carne_77, 
    carne_78, 
    carne_79, 
    carne_80,
    carne_81,
    carne_82,
    carne_83,
    carne_84,
    carne_85,
    carne_86,
    carne_87,
    carne_88,
    carne_89,
    carne_90,
    carne_91,
    carne_92,
    carne_93,
    carne_94,
    carne_95,
    carne_96,
    carne_97,
    carne_98,
    carne_99,
    carne_100,
]

array1 = np.array([0, 0, 0 ,0 ,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0 ,0])
array2 = np.array([0, 0, 0 ,0 ,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0 ,0])
array3 = np.array([0, 0, 0 ,0 ,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0 ,0])
array4 = np.array([0, 0, 0 ,0 ,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0 ,0])
array5 = np.array([0, 0, 0 ,0 ,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0 ,0])

carnes_avalicao = np.concatenate((array1, array2, array3, array4, array5))

modelo = LinearSVC()

# Treinando o modelo
modelo.fit(carnes, carnes_avalicao)

carne_aleatoria = [1, 1, 1, 0]
result = modelo.predict([carne_aleatoria])

print("|---------------------------------|\n")
print("|--Análise de um animal qualquer--|\n")
print("|---------------------------------|\n")

if result == 0:
    print("---> Carne ruim")
else:
    print("---> Carne boa\n")

# Analisando por conjunto e mostrando a acurácia
carne_teste1 = [0, 0, 1, 1]
carne_teste2 = [0, 1, 0, 0]
carne_teste3 = [1, 1, 1, 0]
carne_teste4 = [1, 0, 1, 1]

teste_dados = [
    carne_teste1,
    carne_teste2,
    carne_teste3,
    carne_teste4
]
teste_avaliacao = [0, 0, 1, 1]
previsoes = modelo.predict(teste_dados)

print("|------------------------------------|\n")
print("|-Análise da acurácia de um conjunto-|\n")
print("|------------------------------------|\n")

resultado = (accuracy_score(teste_avaliacao, previsoes))*100
print ("--> A acurácia do sistema é de: {}%\n".format(resultado))

# Como saber se o algoritmo tem uma acuraria boa?
print("|------------------------------------|\n")
print("|---------Avaliando Acurácia---------|\n")
print("|------------------------------------|\n")

dummy_clf = DummyClassifier(strategy="uniform")

dummy_clf.fit(teste_dados, teste_avaliacao)

dummy_clf.predict(teste_dados)

score = dummy_clf.score(teste_dados, teste_avaliacao)
print ("--> A acurácia do sistema é de: {}%".format(score*100))
