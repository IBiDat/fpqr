import fpqr

from sklearn.datasets import make_regression

if __name__ == '__main__':
    x, y, true_beta = make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=2,
                           bias=10.0, noise=2.0, shuffle=True, coef=True, random_state=None)

    fpqr_li = fpqr.FPQRegression(quantile=0.5, n_components=3, metric='li')
    fpqr_li.fit(x, y)

    fpqr_dodge = fpqr.FPQRegression(quantile=0.5, n_components=3, metric='dodge')
    fpqr_dodge.fit(x, y)

    fpqr_choi = fpqr.FPQRegression(quantile=0.5, n_components=3, metric='choi')
    fpqr_choi.fit(x, y)



import fpqr
from sklearn.datasets import make_regression

# Generate a dataset with 1000 observations, 10 predictive variables and 2 response variables.
x, y, true_beta = make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=2,
                                  bias=10.0, noise=2.0, shuffle=True, coef=True, random_state=None)

fpqr_li = fpqr.FPQRegression(quantile=0.5, n_components=3, metric='li')
fpqr_li.fit(x, y)

print(fpqr_li.coef_)
print(fpqr_li.intercept_)

predictions = fpqr_li.predict(x)
