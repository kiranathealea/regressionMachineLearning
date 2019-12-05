import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#read dataset csv
data_year_x = pd.read_csv("maize.csv", usecols = ['Year'], nrows = 54)
x = np.array(data_year_x)
x_list = x.ravel()
data_crop_y = pd.read_csv("maize.csv", usecols = ['Crop'], nrows = 54)
y = np.array(data_crop_y)

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
rmse_ = mean_squared_error(y, y_pred)
r2_ = r2_score(y, y_pred)

# printing metrics of the linear model
print('The RMSE of the linear regression model is {}'.format(rmse_))
print('The R2 score of the linear regression model is {}'.format(r2_))

#prediction for 2015-2020
years = np.array([2015, 2016, 2017, 2018, 2019, 2020])
year = years[:, np.newaxis]
y_next = model.predict(year)
y_next = np.round(y_next,4)


x_final=np.concatenate((x_list, years))
y_final=np.concatenate((y_pred, y_next))
y_final_print=np.concatenate((y, y_next))

x_dot =  years
y_dot = y_next

# baca nama wilayah dan kode
wilayah = pd.read_csv("maize.csv", usecols = ['Entity'], nrows=54)
wilayah = np.array(wilayah)
wilayah = wilayah.ravel()
wil_tambahan = wilayah[0:6]
wilayah=np.concatenate((wilayah, wil_tambahan))
kode = pd.read_csv("maize.csv", usecols=['Code'], nrows=54)
kode = np.array(kode)
kode = kode.ravel()
kode_tambahan = kode[0:6]
kode=np.concatenate((kode, kode_tambahan))

#dictionary untuk write csv
recsv = {
        'Entity' : wilayah,
        'Code' : kode,
        'Year': x_final,
        'crop(tonnes per hectare)': y_final_print.ravel()
        }

df = DataFrame(recsv, columns= ['Entity','Code','Year', 'crop(tonnes per hectare)'])
export_csv = df.to_csv(r'/home/leathea/Downloads/Machine-Learning/PolynomialRegression/test.csv', index = None, header=True) 

#plot grafik
plt.scatter(x, y, s=10, color='c')
plt.scatter(x_dot, y_dot, s=10, color='r')
# plt.scatter(x, y, s=10)
plt.plot(x_final, y_final, color='m')
plt.show()

