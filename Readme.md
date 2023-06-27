# Machine Learning Algorithms from Scratch using NumPy

This repository contains Python implementations of various machine learning algorithms from scratch using NumPy. The implemented algorithms include Naive Bayes, Linear Regression, Adaboost, and K-means. Each algorithm is implemented with specific features and functionalities as described below.

## Algorithms

### 1. Naive Bayes

The Naive Bayes algorithm implemented in this repository includes the following features:

- Underflow protection: The algorithm uses a logarithmic transformation to prevent underflow issues when working with small probabilities.
- Additive smoothing: It incorporates additive smoothing, also known as Laplace smoothing or Lidstone smoothing, to handle unseen features or zero probabilities.
- Numeric attribute support: The algorithm handles numeric attributes by assuming a normal distribution and estimating the mean and standard deviation for each class.

### 2. Linear Regression

The Linear Regression algorithm implemented in this repository includes the following features:

- L2 regularization: It incorporates L2 regularization, also known as Ridge regression, to prevent overfitting by penalizing large coefficients.
- Stochastic Gradient Descent (SGD): It provides an option for using stochastic gradient descent as the optimization algorithm, which randomly selects a subset of data for each iteration.

### 3. Adaboost

The Adaboost algorithm implemented in this repository includes the following features:

- Support for any base algorithm: It can work with any base algorithm or multiple base algorithms, making it versatile for different tasks and models.
- Prediction confidence: The algorithm provides prediction confidence by weighting the predictions from the base algorithms based on their performance.
- Boosting iterations: It iteratively improves the base algorithm by focusing on the misclassified samples from the previous iterations.

### 4. K-means

The K-means algorithm implemented in this repository includes the following features:

- Weighted attributes: It supports weighted attributes, allowing different importance levels for each attribute during clustering.
- Multiple distance functions: The algorithm provides multiple distance functions, such as Euclidean distance, Manhattan distance, and cosine similarity, to cater to different data types and characteristics.
- Repeated runs: It allows the algorithm to be repeated multiple times with different initializations to improve the stability and quality of the clustering results.
- K-means++ initialization: It uses the K-means++ initialization method to select initial cluster centers, improving the convergence speed and quality of the final clusters.
- Cluster performance evaluation: The algorithm includes performance evaluation metrics, such as within-cluster sum of squares (WCSS), to assess the quality of the clustering results.

## Usage

To use any of the implemented algorithms, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/petkostefan/ml-algorithms-from-scratch.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the python files or import the classes

