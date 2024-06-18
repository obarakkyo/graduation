import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###データセットの読みこみ###
data = load_breast_cancer()
X = data.data
y = data.target

#訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#LightGBMデータセットの作成
train_data = lgb.Dataset(x_train, label=y_train)

#ハイパーパラメータの設定
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

#モデルの訓練
num_rounds = 100
bst = lgb.train(params, train_data, num_rounds)

#予測
y_pred = bst.predict(x_test)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

#精度の計算
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')