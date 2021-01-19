# Работа с матрицами
import numpy as np
# Работа с датафреймами + матрица корреляций
import pandas as pd
# Рисование
from matplotlib import pyplot as plt
# Рисование красивой матрицы кореляций
import seaborn as sn
# Алгоритм
from sklearn.decomposition import PCA
# Датасеты
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# # Генерация датасета с 60 признаками, подсчет его матрицы корелляций
# rs1 = np.random.RandomState(0)
# rs1_data = pd.DataFrame(rs1.rand(10, 60))
# rs1_data.T.to_csv('out.csv', index = False, header=True)
# rs1_corr = rs1_data.corr()

# # Генерация датасета с 1000 признаками, подсчет его матрицы корреляций
# rs2 = np.random.RandomState(13)
# rs2_data = pd.DataFrame(rs2.rand(10, 1000))
# rs2_corr = rs2_data.corr()

# Загрузка датасета ирисы и подсчет его матрицы корреляций
iris = load_iris()
irisDf = pd.DataFrame(iris.data)
irisDf.to_csv('iris.csv', index=False, header=True)
irisCorr = irisDf.corr()

# Загрузка датасета рак и подсчет его матрицы корреляций
# cancer = load_breast_cancer()
# cancerDf = pd.DataFrame(cancer.data)
# cancerDf.to_csv('cancer.csv', index = False, header=True)
# cancerCorr = cancerDf.corr()

# Составление списка со слабой корелляцией
corrSum = [[i, 0] for i in range(irisCorr.shape[0])]
print("Correlation sum\n{}\n".format(corrSum))
for i in range(irisCorr.shape[0]):
    for j in range(irisCorr.shape[1]):
        corrSum[i][1] += abs(irisCorr[i][j])

corrSum.sort(key=lambda x: x[1], reverse=True)
print("Sorted correlation sum\n{}\n".format(corrSum))

# Рисуем матрицу кореляций
f1 = plt.figure("Correlation matrix")
sn.heatmap(irisCorr, annot=True)
# plt.matshow(cancerCorr)
plt.title("Correlation matrix")

# Создание модели и выполнение алгоритма на датасете
sklearn_pca = PCA(n_components=1)
sklearn_transf = sklearn_pca.fit_transform(irisDf)

# Получаем значения описываемой дисперсии
variance = sklearn_pca.explained_variance_ratio_
# Ковариации
covariation = sklearn_pca.get_covariance()
# Собственных значений
eigenValue = sklearn_pca.explained_variance_
# Собственных векторов
eigenVector = sklearn_pca.components_

print("Eigen vectors\n{}".format(eigenVector))
print("Eigen values\n{}".format(eigenValue))
print("explained variance ratio\n{}".format(variance))
print("covariation matrix\n{}".format(covariation))

print("Kaiser rule says that we should take those PC which eigen values are greater that 1\n{}".format(
    eigenValue))

# Правило сломанной трости
broken_stick = []
for i in range(len(variance)):
    broken_stick.append(sum(1 / (i + 1) for i in range(len(variance) - i)) / len(variance))
print("Broken stick rule\n{}\n{}".format(broken_stick, variance))
print("We take those PC, that are greater than corresponding pieces of stick")

# Правило каменистой осыпи
f2 = plt.figure("Scree plot")
plt.plot([i + 1 for i in range(len(variance))], variance)
plt.scatter([i + 1 for i in range(len(variance))], variance, marker="o")
plt.xlabel("Principal component number")
plt.ylabel("Variation")

# Факторная нагрузка
PCAFeatures = pd.DataFrame(sklearn_transf, columns=['PC1'])
print(irisDf, '\n', PCAFeatures)
temp = irisDf.T.append(PCAFeatures.T)
print(temp)
loadingMatrix = temp.T.corr()
f3 = plt.figure("Factor analysis")
sn.heatmap(loadingMatrix[-1:], annot=True)
plt.title("Loading matrix")

for i in range(1):
    print('PC' + str(i + 1), sum(abs(loadingMatrix['PC' + str(i + 1)][:])) - 1)

# # Отображение варианта Scikit-learn
f4 = plt.figure("Scikit-learn PCA")
plt.scatter(sklearn_transf, range(len(sklearn_transf)), c=iris.target)
plt.title('Метод главных компонент от Sckit-learn')

plt.show()
