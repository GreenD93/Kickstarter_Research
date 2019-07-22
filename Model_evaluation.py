from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time


class Model_evaluation:
    def __init__(self,data,label,scaler='std'):
        shuffled_data = shuffle(data)

        self.x = shuffled_data.drop([label],axis=1)
        self.y = shuffled_data[[label]]

        if scaler == 'minmax':
            scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
            scaler.fit(self.x)
        else:
            scaler = StandardScaler()
            scaler.fit(self.x)
        X_scaled = pd.DataFrame(scaler.transform(self.x), columns=self.x.columns)



        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.33,
                                                                                random_state=42)


    def random_forest(self):
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(self.x_train, self.y_train)
        print("RF :", accuracy_score(clf.predict(self.x_test), self.y_test))
        print(classification_report(self.y_test, clf.predict(self.x_test)))
        scores = cross_val_score(clf, self.x, self.y, cv=5)
        print(scores)

    def logistic_regression(self):
        print('logistic_regression')
        logreg = linear_model.LogisticRegression(C=2.0, random_state=42, solver='sag', multi_class='multinomial',
                                                 warm_start=True)
        logreg.fit(self.x_train, self.y_train)
        print("LR :", accuracy_score(logreg.predict(self.x_test), self.y_test))
        print(classification_report(self.y_test, logreg.predict(self.x_test)))
        scores = cross_val_score(logreg, self.x, self.y, cv=5)
        print(scores)

    def svm(self):
        result = LinearSVC(random_state=0).fit(self.x_train, self.y_train)
        print("SVM :", accuracy_score(result.predict(self.x_test), self.y_test))
        print(classification_report(self.y_test, result.predict(self.x_test)))
        scores = cross_val_score(result, self.x, self.y, cv=5)
        print(scores)


    def mlp(self):

        mlp_clf = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=35,
            alpha=1e-4,
            solver='sgd',
            verbose=10,
            tol=1e-4,
            random_state=1,
            learning_rate_init=.1
        )

        start = time.time()
        mlp_clf.fit(self.x_train, self.y_train)
        end = time.time()
        print('Time: {:f}s'.format(end - start))
        y_pred = mlp_clf.predict(self.x_test)
        print("테스트 정확도: {:.3f}".format(accuracy_score(y_pred, self.y_test)))
        print(classification_report(self.y_test, mlp_clf.predict(self.x_test)))
