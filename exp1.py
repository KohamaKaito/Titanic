# 特徴量の相関関係


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv('input/train.csv')

encoder = LabelEncoder()
df_train['Name'] = encoder.fit_transform(df_train['Name'].values)
df_train['Sex'] = encoder.fit_transform(df_train['Sex'].values)
df_train['Ticket'] = encoder.fit_transform(df_train['Ticket'].values)


df_corr = df_train.corr()
print(df_corr)


sns.heatmap(df_corr, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1)
plt.show()
