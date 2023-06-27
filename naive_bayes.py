import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveBayes():
    
    def __init__(self, smoothing=1):
        self.smoothing = smoothing

    def learn(self, X:pd.DataFrame, y:pd.Series):
        
        self.model = {}

        apriori = y.value_counts()
        apriori /= apriori.sum()
        self.model['apriori'] = apriori

        # Calculate probabilities for categorical data
        for attribute in X.select_dtypes('object').columns:
            mat_cont = pd.crosstab(X[attribute], y)
            mat_cont = (mat_cont + self.smoothing) / (
                mat_cont.sum(axis=0) + self.smoothing * X[attribute].nunique())
            self.model[attribute] = mat_cont

        # Calculate probabilities for numeric data
        for attribute in X.select_dtypes(np.number).columns:
            mat_cont = pd.crosstab(X[attribute], y)
            self.model[attribute] = {}
            for class_value in self.model['apriori'].index:
                self.model[attribute][class_value] = {}
                self.model[attribute][class_value]['mean'] = X[y==class_value][attribute].mean()
                self.model[attribute][class_value]['std'] = X[y==class_value][attribute].std()

    def predict(self, X:pd.DataFrame):

        predictions = []
            
        for i in range(len(X)):
        
            class_probabilities = {}

            for class_value in self.model['apriori'].index:
                probability = 1

                for attribute in self.model:
                    if attribute == 'apriori':
                        probability += np.log(self.model[attribute][class_value])
                    elif X[attribute].dtype == 'object':
                        probability += np.log(self.model[attribute][class_value][X.iloc[i][attribute]])
                    else:
                        probability += np.log(
                            norm.pdf(X.iloc[i][attribute],
                                     self.model[attribute][class_value]['mean'],
                                     self.model[attribute][class_value]['std']))
                class_probabilities[class_value] = probability

            prediction = max(class_probabilities, key=class_probabilities.get)
            predictions.append(prediction)

        return predictions


if __name__ == '__main__':
    df = pd.read_csv('data/drug.csv')
    X = df.drop('Drug', axis=1)
    y = df['Drug']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = NaiveBayes()
    model.learn(X_train, y_train)

    pred = model.predict(X_test)
    print(pred)
    print(accuracy_score(y_test, pred))
