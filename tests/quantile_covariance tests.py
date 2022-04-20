import fpqr

from sklearn.datasets import make_regression

if __name__ == '__main__':
    x, y, true_beta = make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=1,
                           bias=10.0, noise=2.0, shuffle=True, coef=True, random_state=None)

    qcov = fpqr.QuantileCovariance(quantile=0.5, metric='li')
    r = qcov.fit(x, y)

    qcov = fpqr.QuantileCovariance(quantile=0.5, metric='choi')
    r = qcov.fit(x, y)

    qcov = fpqr.QuantileCovariance(quantile=0.5, metric='dodge')
    r = qcov.fit(x, y)