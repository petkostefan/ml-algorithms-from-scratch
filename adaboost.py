import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class AdaBoost():
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def learn(self, X:pd.DataFrame, y:pd.Series, algorithm=[GaussianNB], lr=1):

        self.algorithm = algorithm

        N, M = X.shape
        alphas = pd.Series(np.array([1/N]*N), index=X.index)

        self.ensemble = []
        self.weights = np.zeros(self.n_estimators)

        for t in range(self.n_estimators):
            
            # Pick random algorithm from passed list and call it
            alg = np.random.choice(self.algorithm, 1)[0]()

            model = alg.fit(X, y, sample_weight=alphas)
            predictions = model.predict(X)
            error = (predictions!=y).astype(int)
            weighted_error = (error*alphas).sum()
            w = 1/2 * np.log((1-weighted_error) / weighted_error)

            self.ensemble.append(model)
            self.weights[t] = w

            # Calculating new alphas
            alphas *= np.exp(-w * predictions * y * lr)

            # Normalization
            alphas /= alphas.sum()

    def predict(self, X:pd.DataFrame):
        predictions = pd.DataFrame([model.predict(X) for model in self.ensemble]).T
        final = np.sign(predictions.dot(self.weights))
        confidence = predictions.apply(
            pd.Series.value_counts, axis=1).fillna(0) # Count predictions by class
        confidence /= self.n_estimators # Normalize
        confidence = confidence.max(axis=1) # Save greater value (predicted class probability)

        return final, confidence


if __name__ == '__main__':

    df = pd.read_csv('data/drugY.csv')
    X = df.drop('Drug',axis=1)
    y = df['Drug']*2-1 # Transform target variable from 0,1 to -1,1
    X = pd.get_dummies(X)

    model = AdaBoost()
    model.learn(X, y)
    pred, confidence = model.predict(X)
    print(accuracy_score(y, pred))
    print(pd.DataFrame({'Prediction': pred, 'Confidence': confidence}))
