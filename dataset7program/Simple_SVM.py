from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def main():
    status_dict = {
        "target_csv" : f"../CSV/dataset7CSV/doc2vec/2label.csv"
    }

    # データセットのロード #
    df = pd.read_csv(status_dict["target_csv"], index_col=0)
    print(f"探索対象のデータ = {status_dict['target_csv']}")

    # データの分割 #
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)
    print(f"=======分割後=======")
    print(f"x_train = {x_train.shape}")
    print(f"x_test  = {x_test.shape}")
    print(f"y_train = {y_train.shape}")
    print(f"y_test  = {y_test.shape}")


    # ハイパーパラメータの候補を設定 #
    param_grid = {
        'C' : [0.1, 1, 10, 100],
        'kernel' : ['linear', 'rbf'],
        'gamma' : ['scale', 'auto'],
    }

    # GridSearchCVを使用して最適なモデルを探索 #
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="accuracy")
    grid_search.fit(x_train, y_train)

    # 最適なモデルでテストセットを予測 #
    y_pred = grid_search.best_estimator_.predict(x_test)
    

    # 正解率を計算 #
    accuracy = accuracy_score(y_test, y_pred)

    # print(f"Best Parameters : {grid_search.best_params_}")
    print(f"Accuracy : {accuracy}")


if __name__ == "__main__":
    main()