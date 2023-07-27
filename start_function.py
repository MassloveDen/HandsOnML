import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# Загрузить данные
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=', ')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=', ',
                             delimiter='\t', encoding='latin1', na_values="n/a")
# Подготовить данные
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
Х = np.c_[country_stats["GDP per capita"]]
у = np.c_[country_stats["Life satisfaction"]]
# Визуализировать данные
country_stats.plot(kind='scatter', x="GDP per capita",
                   y='Life satisfaction')
plt.show()
# Выбрать линейную модель
model = sklearn.linear_model.LinearRegression()
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# Обучить модель
model.fit(Х, у)
# Выработать прогноз для КИпра
X_new = [[22587]]
print(model.predict(X_new))
# ВВП на душу населения КИпра
# выводит [ [ 5. 9 6242338] 1
