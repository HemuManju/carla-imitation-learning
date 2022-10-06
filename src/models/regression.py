from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
import matplotlib.pyplot as plt

from src.data.create_data import create_regression_data


def least_squares(dataset):

    steering, features = create_regression_data(dataset['training'])
    regr = LinearRegression(fit_intercept=False)
    regr.fit(features, steering)

    true_steering, test_features = create_regression_data(dataset['validation'])
    predited_steering = regr.predict(test_features)

    mse = mean_squared_error(true_steering, predited_steering, squared=False)
    print(mse)

    plt.plot(predited_steering, 'o')
    plt.plot(true_steering, '*')
    plt.show()

    print(regr.coef_)

    return None
