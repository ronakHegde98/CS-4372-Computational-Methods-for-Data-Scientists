import pandas as pd
from preprocess import preprocessor
from NeuralNet import NeuralNet

if __name__ == "__main__":
  dataset_url = "https://raw.githubusercontent.com/ronakHegde98/CS-4372-Computational-Methods-for-Data-Scientists/master/data/diabetic_data.csv"
  df = pd.read_csv(dataset_url)
  X_train, X_test, y_train, y_test = preprocessor(df)
  print(X_train.head())

  