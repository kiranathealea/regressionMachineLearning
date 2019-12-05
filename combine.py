import os
import glob
import pandas as pd
#set working directory
os.chdir("/home/leathea/Downloads/Machine-Learning/PolynomialRegression/hasilmaize")

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_maize.csv", index=False, encoding='utf-8-sig')