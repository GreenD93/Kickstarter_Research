import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from  sklearn  import  linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class feature_importance:
    def __init__(self,data,label,scaler='std'):
        shuffled_data = shuffle(data)

        self.scaler = scaler
        self.x = shuffled_data.drop([label],axis=1)
        self.y = shuffled_data[[label]]
        self.feature_name = self.x.columns.tolist()

        minmax_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        minmax_scaler.fit(self.x)
        self.minmax_x = pd.DataFrame(minmax_scaler.transform(self.x), columns=self.x.columns)

        std_scaler = StandardScaler()
        std_scaler.fit(self.x)
        self.std_x = pd.DataFrame(std_scaler.transform(self.x), columns=self.x.columns)


    # 음수 값이 없어야함. Min-max scaler
    def chisquare(self):

        x = self.minmax_x

        chi2_selector = SelectKBest(chi2, k=len(self.feature_name))
        chi2_selector.fit(x, self.y)
        # Look at scores returned from the selector for each feature
        chi2_scores = pd.DataFrame(list(zip(self.feature_name, chi2_selector.scores_, chi2_selector.pvalues_)), columns=['ftr', 'score', 'pval'])

        # you can see that the kbest returned from SelectKBest
        #+ were the two features with the _highest_ score
        kbest = np.asarray(self.feature_name)[chi2_selector.get_support()]

        feature_dict = dict(zip(self.feature_name,chi2_selector.scores_))
        ranking_list = reversed(sorted(feature_dict.items(), key =lambda feature_dict:feature_dict[1]))
        print('Feature ranking:')
        for num, feature in enumerate(ranking_list):
            print('Rank : %d \t Feature : %s (%f)' % (num + 1, feature[0], feature[1]))

    # 음수 값이 없어야함. Standard scaler 또는 min_max scaler 가능!
    def random_forest(self):
        if self.scaler == 'std':
            x = self.std_x
        else:
            x = self.minmax_x

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(x, self.y)

        importances = clf.feature_importances_
        std = np.std([clf.feature_importances_ for tree in clf.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for num, f in enumerate(indices):
            print('Rank : %d \t Feature : %s (%f)' % (num, self.feature_name[f], importances[f]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.x.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.x.shape[1]), indices)
        plt.xlim([-1, self.x.shape[1]])
        plt.show()

    def logistic_regression(self):
        if self.scaler == 'std':
            x = self.std_x
        else:
            x = self.minmax_x

        logreg = linear_model.LogisticRegression(C=2.0, random_state=42, solver='sag', multi_class='multinomial',
                                                 warm_start=True)
        logreg.fit(x, self.y)

        feature_importance = abs(logreg.coef_[0])
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)

        feature_dict = dict(zip(self.feature_name, feature_importance))
        ranking_list = reversed(sorted(feature_dict.items(), key=lambda feature_dict: feature_dict[1]))
        print('Feature ranking:')
        for num, feature in enumerate(ranking_list):
            print('Rank : %d \t Feature : %s (%f)' % (num + 1, feature[0], feature[1]))

        pos = np.arange(sorted_idx.shape[0]) + .5

        featfig = plt.figure()
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, feature_importance[sorted_idx], align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(self.x.columns)[sorted_idx], fontsize=8)
        featax.set_xlabel('Relative Feature Importance')

        plt.tight_layout()
        plt.show()