import pandas as pd
import numpy as np
from pandas import DataFrame

data_year_x = pd.read_csv("maize.csv", usecols = ['Year'], skiprows = range(1, 55))
Year = np.array(data_year_x)
data_crop_y = pd.read_csv("maize.csv", usecols = ['Crop'], skiprows = range(1, 55))
Crop = np.array(data_crop_y)
print(Year)
# print(Year, Crop)

Year1 = Year.ravel()
Crop1 = Crop.ravel()
# print(Year1)

wilayah = pd.read_csv("maize.csv", usecols = ['Entity'], skiprows = range(1, 55))
wilayah = np.array(wilayah)
wilayah = wilayah.ravel()

kode = pd.read_csv("maize.csv", usecols=['Code'], skiprows = range(1, 55))
kode = np.array(kode)
kode = kode.ravel()


# recsv = {'Wilayah': wilayah,
#         'Kode': kode,
#         'Year': Year1,
#         'Crop': Crop1
#         }
# df = DataFrame(recsv, columns= ['Wilayah', 'Kode', 'Year', 'Crop'])
# export_csv = df.to_csv(r'/home/leathea/Downloads/Machine-Learning/test.csv', index = None, header=True) 