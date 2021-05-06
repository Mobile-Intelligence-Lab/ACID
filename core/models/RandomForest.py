from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self, n_estimators=200):
        super(RandomForest, self).__init__()
        self.cls = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, X, y):
        self.cls.fit(X, y)

    def predict(self, x):
        return self.cls.predict(x)
