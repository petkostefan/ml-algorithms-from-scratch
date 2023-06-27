import pandas as pd
import numpy as np
import time


class LinearRegression():

    def __init__(self, iterations=200, lr=0.2):
        self.iterations = iterations
        self.lr = lr

    def learn(self, X: pd.DataFrame, y:pd.Series, normalize=True, L=0.1,
              stohastic=True, adaptive_lr=False, weights=None) -> None:
        """
        Fit model
        """
        self.X_mean = X.mean()
        self.X_std = X.std()

        # Normalize features
        if normalize:
            X = (X - self.X_mean) / self.X_std

        X.insert(0, 'X0', 1)

        # Stohastic gradient descent
        if stohastic:
            X['y'] = y
            X = X.sample(frac=1).reset_index(drop=True) # Shuffle samples
            y = X[['y']]
            X.drop('y', axis=1, inplace=True)

        self.M, self.N = X.shape
        X = X.to_numpy()
        y = y.to_numpy()

        # Transfer learning if initial weights are provided
        if weights:
            self.W = weights

        # If not, initialize to random values
        else:
            self.W = np.random.random((1, self.N))

        # Run model optimization for given number of iterations
        for i in range(self.iterations):

            # If stohastic, optimize on single value
            if stohastic:
                pred = X[i%self.M].dot(self.W.T)
                error = pred - y[i%self.M]
                grad = error * X[i%self.M] / self.M
                W_n = self.W.copy()
                W_n[0][0] = 0
                grad += 2*L*W_n[0]

            # Otherwise, optimize on the whole dataset
            else:            
                pred = X.dot(self.W.T)
                error = pred - y
                grad = error.T.dot(X) / self.M
                W_n = self.W.copy()
                W_n[0][0] = 0       
                grad += 2*L*W_n

            # Modify learning rate
            if adaptive_lr:
                self.lr = self._exp_decay(i, self.lr)

            # Calculate new weights
            self.W = self.W - self.lr * grad

            # Calculate and print MSE and gradient
            MSE = error.T.dot(error) / self.M
            grad_norm = abs(grad).sum()
            print(f'I: {i}, Gradient: {grad_norm:.2f}, MSE: {MSE} , LR: {self.lr:.3f}')

            # Break if gradient is below a given threshold
            if grad_norm < 0.001: break

    def predict(self, X: pd.DataFrame, normalize=True):
        """
        Generate predictions
        """

        if normalize:
            X = (X - self.X_mean) / self.X_std

        X.insert(0, 'X0', 1)
        X = X.to_numpy()

        return X.dot(self.W.T)

    def _exp_decay(self, i, lr):
        """
        Adaptive exponential learning rate
        """
        
        k = 1E-4
        lr = lr * np.exp(-k*i)
        return lr


if __name__ == '__main__':
    df = pd.read_csv('data/boston.csv')

    X = df.drop('MEDV', axis=1)
    y = df[['MEDV']]

    model = LinearRegression()
    model.learn(X,y)
    pred = model.predict(X)

    y = y.to_numpy()

    mse = (pred-y).T.dot(pred-y) / len(y)
    rmse = np.sqrt(((pred-y).T.dot(pred-y) / len(y)))[0][0]
    print(f'RMSE: {rmse:.2f}, R2: {1-rmse/y.mean():.2f}')
    # print(model.W)
