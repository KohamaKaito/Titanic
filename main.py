import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# train.csv,test.csvを読み込む
df_train = pandas.read_csv('input/train.csv')
df_test = pandas.read_csv('input/test.csv')


# 必要な項目を抽出する
df_train = df_train[['Survived', 'Pclass', 'Sex', 'Fare']]
df_test = df_test[['PassengerId','Pclass', 'Sex', 'Fare']]


# 性別をLabelEncoderを利用して数値化する
encoder_sex = LabelEncoder()
df_train['Sex'] = encoder_sex.fit_transform(df_train['Sex'].values)
df_test['Sex'] = encoder_sex.fit_transform(df_test['Sex'].values)


# 欠損データの補間
df_test = df_test.fillna(df_test.median())


# 数値を標準化する
standard = StandardScaler()
df_train_std = pandas.DataFrame(standard.fit_transform(df_train[['Pclass', 'Fare']]), columns=['Pclass', 'Fare'])
df_test_std = pandas.DataFrame(standard.fit_transform(df_test[['Pclass', 'Fare']]), columns=['Pclass', 'Fare'])


# Fare を標準化
df_train['Pclass'] = df_train_std['Pclass']
df_train['Fare'] = df_train_std['Fare']
df_test['Pclass'] = df_test_std['Pclass']
df_test['Fare'] = df_test_std['Fare']


# データを分割
x_train = df_train.drop(columns='Survived')
y_train = df_train[['Survived']]
x_test = df_test.drop(columns='PassengerId')
df_test_index = df_test[['PassengerId']]


# 学習モデルの設定
model = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    min_samples_split=2,
    min_samples_leaf=4,
    bootstrap=True)  


# 学習
model.fit(x_train, y_train)
y_test = model.predict(x_test)


# 提出用ファイルの生成
df_output = pandas.concat([df_test_index, pandas.DataFrame(y_test, columns=['Survived'])], axis=1)
df_output.to_csv('result.csv', index=False)