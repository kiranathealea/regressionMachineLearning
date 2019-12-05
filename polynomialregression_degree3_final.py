import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import os
import glob

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

daftar_kota = ['Bandung', 'Bantul', 'Bekasi', 'Bogor', 'Ciamis', 'Cianjur', 'Cirebon', 'DIY', 'Garut', 'Gunung Kidul', 'Indramayu', 'Karawang', 'Kulon Progo', 'Kuningan', 'Majalengka', 'Purwakarta', 'Sleman', 'Subang', 'Sukabumi', 'Sumedang', 'Tasikmalaya']
i=0
for i in range (19):
    # folder = '/home/leathea/Downloads/Machine-Learning/PolynomialRegression/maize_csv_perdaerah/'+daftar_kota[i]+".csv"
    folder = '/home/leathea/Downloads/Machine-Learning/PolynomialRegression/tomato_csv_perdaerah/'+daftar_kota[i]+".csv"
    #read dataset csv
    data_year_x = pd.read_csv(folder, usecols = ['Year'])
    x = np.array(data_year_x)
    x_list = x.ravel()
    data_crop_y = pd.read_csv(folder, usecols = ['Crop'])
    y = np.array(data_crop_y)

    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    # print(x_poly)
    y_poly_pred = model.predict(x_poly)
    # theta = model.coef_
    # print(model.coef_)
    rmse_ = mean_squared_error(y, y_poly_pred)
    r2_ = r2_score(y, y_poly_pred)

    # printing metrics of the linear model
    print('The RMSE of the linear regression model is {}'.format(rmse_))
    print('The R2 score of the linear regression model is {}'.format(r2_))

    #prediction for 2015-2020
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020])
    year = years[:, np.newaxis]
    year = polynomial_features.fit_transform(year)
    y_next = model.predict(year)
    y_next = np.round(y_next,4)


    x_final=np.concatenate((x_list, years))
    y_final=np.concatenate((y_poly_pred, y_next))
    y_final_print=np.concatenate((y, y_next))

    x_dot =  years
    y_dot = y_next

    # baca nama wilayah dan kode
    wilayah = pd.read_csv(folder, usecols = ['Entity'])
    wilayah = np.array(wilayah)
    wilayah = wilayah.ravel()
    wil_tambahan = wilayah[0:6]
    wilayah=np.concatenate((wilayah, wil_tambahan))
    kode = pd.read_csv(folder, usecols=['Code'])
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
    # export_csv = df.to_csv(r'/home/leathea/Downloads/Machine-Learning/PolynomialRegression/hasilmaize/poly2/'+daftar_kota[i]+r'_maize.csv', index = None, header=True)
    export_csv = df.to_csv(r'/home/leathea/Downloads/Machine-Learning/PolynomialRegression/hasiltomato/poly2/'+daftar_kota[i]+r'_tomato.csv', index = None, header=True) 

    # #plot grafik
    # plt.scatter(x, y, s=10, color='c')
    # plt.scatter(x_dot, y_dot, s=10, color='r')
    # # plt.scatter(x, y, s=10)
    # plt.plot(x_final, y_final, color='m')
    # plt.show()
else:
    # os.chdir("/home/leathea/Downloads/Machine-Learning/PolynomialRegression/hasilmaize/poly2")
    os.chdir("/home/leathea/Downloads/Machine-Learning/PolynomialRegression/hasiltomato/poly2")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    # combined_csv.to_csv( "/home/leathea/Downloads/Machine-Learning/PolynomialRegression/predicted_poly2_maize.csv", index=False, encoding='utf-8-sig')
    combined_csv.to_csv( "/home/leathea/Downloads/Machine-Learning/PolynomialRegression/predicted_poly2_tomato.csv", index=False, encoding='utf-8-sig')
    print('all done')
