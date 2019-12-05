import numpy as np
import pandas as pd

daftar_kota = ['Bandung', 'Bantul', 'Bekasi', 'Bogor', 'Ciamis', 'Cianjur', 'Cirebon', 'DIY', 'Garut', 'Gunung Kidul', 'Indramayu', 'Karawang', 'Kulon Progo', 'Kuningan', 'Majalengka', 'Purwakarta', 'Sleman', 'Subang', 'Sukabumi', 'Sumedang', 'Tasikmalaya']
df = pd.read_csv("tomato.csv")
df_entity = df.groupby('Entity')
i = 0
for i in range(19):
    group = df_entity.get_group(daftar_kota[i])
    # export_csv = group.to_csv(r'/home/leathea/Downloads/Machine-Learning/PolynomialRegression/maize_csv_perdaerah/'+str(daftar_kota[i])+r'.csv', index = None, header=True)
    export_csv = group.to_csv(r'/home/leathea/Downloads/Machine-Learning/PolynomialRegression/tomato_csv_perdaerah/'+str(daftar_kota[i])+r'.csv', index = None, header=True)
else:
    print('all done')
# print(group)

