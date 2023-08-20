import pandas as pd

path = '/home/dimeng/packages/uniclust30_2018_08/uniclust30_2018_08.tsv'
df_uniclust30 = pd.read_csv(path, header=None, sep='\t', usecols=[0])

print(f'length: {len(set(df_uniclust30.iloc[:, 0]))}')