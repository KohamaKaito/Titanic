# ハイパーパラメータのチューニング

import numpy 
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# train.csvを読み込む
df = pandas.read_csv('input/train.csv')


# 必要な項目を抽出する
df = df[['Survived', 'Pclass', 'Sex', 'Fare']]


# ラベル（名称）を数値化する
encoder_sex = LabelEncoder()
df['Sex'] = encoder_sex.fit_transform(df['Sex'].values)


# 数値を標準化する
standard = StandardScaler()
df_std = pandas.DataFrame(standard.fit_transform(df[['Pclass', 'Fare']]), columns=['Pclass', 'Fare'])
df['Pclass'] = df_std['Pclass']
df['Fare'] = df_std['Fare']


# トレーニングデータとテストデータを分ける
x = df.drop(columns='Survived')
y = df[['Survived']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True)
y_train = numpy.ravel(y_train)
y_test = numpy.ravel(y_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# グリッドサーチで LogisticRegression のパラメータを試す
pipe_svc = RandomForestClassifier(random_state=1)

param_grid = {'criterion':['gini','entropy'],
              'n_estimators':[25, 100, 500, 1000, 2000],
              'min_samples_split':[0.5, 2,4,10],
              'min_samples_leaf':[1,2,4,10],
              'bootstrap':[True, False]
              }

grid = GridSearchCV(estimator=RandomForestClassifier(random_state=1), param_grid=param_grid)
grid = grid.fit(x_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
