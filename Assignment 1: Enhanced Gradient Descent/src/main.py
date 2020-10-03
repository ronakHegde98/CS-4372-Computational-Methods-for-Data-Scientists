import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
from linear_model import LinearRegression

def preprocessor(df):
  '''preprocessing our data'''

  #removed after correlation analysis, reasearch into feature names, and paper reading
  drop_columns = ['subject#', 'age', 'sex','test_time', 'total_UPDRS','Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA']

  target_feature = 'motor_UPDRS'

  #drop our columns 
  Y_feature = df[target_feature]
  X_features = df.drop(columns = drop_columns + [target_feature], inplace=True)
  X_features = df.copy(deep=True)

  X_train, X_test, y_train, y_test = train_test_split(X_features,Y_feature, test_size=0.2, random_state = 42)

  #Min-Max Scaling to get [0,1] range for each feature (useful for ML)
  mm_scaler = MinMaxScaler()
  X_train = pd.DataFrame(mm_scaler.fit_transform(X_train), columns = X_train.columns) 
  X_test = pd.DataFrame(mm_scaler.fit_transform(X_test), columns = X_test.columns)

  #added dummy feature for bias term 
  X_train['x1'] = np.ones((len(X_train),1))
  X_test['x1'] = np.ones((len(X_test),1))

  return (X_train, X_test, y_train, y_test)

def sklearn_sgd(X_train, X_test, y_train, y_test):
  model = SGDRegressor(max_iter = 1000, loss = 'squared_loss')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  print("\nSklearn Output for Stochastic Gradient Descent: ")
  print('Model Coefficients: ', model.coef_)
  return y_pred


def print_metrics(y_test, y_pred):
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Linear Regression Assignment')
  parser.add_argument('--version',help=' sklearn or scratch')
  parser.add_argument('--momentum',type=bool, help="True is gradient descent with momentum")

  args = parser.parse_args()
  version = args.version
  momentum = args.momentum

  url = 'https://raw.githubusercontent.com/ronakHegde98/CS-4372-Computational-Methods-for-Data-Scientists/master/parkinsons.data'
  df = pd.read_csv(url)
  X_train, X_test, y_train, y_test = preprocessor(df)

  if(version.lower() == 'sklearn'):
    y_pred = sklearn_sgd(X_train, X_test, y_train,y_test)

  else:
    if(momentum == True):
      momentum_model = LinearRegression(iterations = 1000, learning_rate = 1, momentum=True)
      momentum_model.fit(X_train, y_train)
      y_pred = momentum_model.predict(X_test)
      print('Model Coefficients: ', momentum_model.weights)
    else: 
      model = LinearRegression(iterations = 1000, learning_rate = 0.01, momentum=False)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      print('Model Coefficients: ', model.weights)
  
  print_metrics(y_test, y_pred)

     