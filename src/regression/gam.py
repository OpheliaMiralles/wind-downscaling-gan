import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines


def normalizer(data):
    return (data - np.nanmean(data)) / np.nanstd(data)


def positive_normalizer(data):
    return (data - data.min()) / (data.max() - data.min())


if __name__ == '__main__':
    data_path = pathlib.Path(
        '/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months.csv')
    data = pd.read_csv(data_path)
    data = data.dropna(subset=['u10_hr', 'v10_hr'])
    u10 = data['u10_hr']
    v10 = data['v10_hr']
    covariates = data[['u10', 'v10', 'hour', 'tpi_500', 'aspect', 'blh', 'fsr', 'sp', 'z']]
    normalized_covariates = covariates.copy()
    for col in covariates:
        if np.min(covariates[col]) >= 0:
            normalized_covariates[col] = positive_normalizer(normalized_covariates[col])
        else:
            normalized_covariates[col] = normalizer(normalized_covariates[col])
    normalized_covariates['intercept'] = 1.
    mod = sm.OLS(normalizer(u10), normalized_covariates)
    res = mod.fit()
    print(res.summary())

    bs = BSplines(covariates[['u10', 'v10']], df=[4, 4], degree=[3, 3])
    gam_bs = GLMGam.from_formula('u10_hr ~ u10 + v10 + tpi_500 + hour + aspect + blh + fsr + sp + z + tpi_500:hour',
                                 data=data, smoother=bs)
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    data['hour_range'] = pd.cut(data['hour'], bins = [0,6,12,18,24])
    data['altitude_range'] = pd.qcut(data['altitude_m'], 4)
    for ar in data['altitude_range'].unique():
        sns.pairplot(data[data['altitude_range'] == ar], hue="hour_range",
                     vars= ['u10_hr', 'v10_hr', 'u10', 'v10', 'blh', 'fsr', 'sp', 'z'])
        plt.savefig(f'/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/plots/pairwise_corr_{ar}.png')

    for h in data['hour_range'].unique()[2:]:
        sns.pairplot(data[data['hour_range'] == h], hue="altitude_range",
                     vars=['u10_hr', 'v10_hr', 'u10', 'v10', 'blh', 'fsr', 'sp', 'z'])
        plt.savefig(f'/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/plots/pairwise_corr_{h}.png')
