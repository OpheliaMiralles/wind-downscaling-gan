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
    data = data.assign(wind_speed=lambda x: np.sqrt(x['u10'] ** 2 + x['v10'] ** 2)).assign(
        wind_speed_hr=lambda x: np.sqrt(x['u10_hr'] ** 2 + x['v10_hr'] ** 2))
    data['u10_hr'] = -data['u10_hr']
    to_concat = []
    for s, d in data.groupby('station'):
        d = d[(d['wind_speed_hr']> 0) & (d['wind_speed']>0)]
        u10_hr = d['u10_hr']
        ws_hr = np.log(d['wind_speed_hr'])
        d = d.assign(log_wind_speed = lambda x: np.log(x['wind_speed']))
        covariates = d[['u10', 'fsr', 'z']] #'tpi_500', 'aspect', sp, hour, blh
        normalized_covariates = covariates.copy()
        for col in covariates:
            if col in ['tpi_500', 'aspect']:
                pass
            elif np.min(covariates[col]) >= 0 and col != 'hour':
                normalized_covariates[col] = positive_normalizer(normalized_covariates[col])
            else:
                normalized_covariates[col] = normalizer(normalized_covariates[col])
        normalized_covariates['intercept'] = 1.
        mod = sm.OLS(normalizer(u10_hr), normalized_covariates)
        res = mod.fit()
        to_concat.append(pd.DataFrame([[s, d['altitude_m'].unique()[0], res.rsquared, res.rsquared_adj, res.aic, res.bic]],
                                      columns=['station', 'altitude_m', 'Rsquared', 'Rsquared_adj', 'AIC', 'BIC']))
    res_ols = pd.concat(to_concat)
    res_ols = res_ols.sort_values('altitude_m').set_index('altitude_m')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
    for col, color in zip([c for c in res_ols.columns if c !='station'], ['navy', 'royalblue', 'salmon', 'chocolate']):
        if col in ['AIC', 'BIC']:
            ax = ax2
        else:
            ax=ax1
        ax.scatter(res_ols.index, res_ols[col],
                s=8, color=color, marker='x', label=col)
        ax.set_xlabel('station altitude (m)')
    ax1.legend()
    ax2.legend()
    fig.suptitle('OLS regression results per station')
    plt.savefig('/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/plots/ols_results.png')


    bs = BSplines(covariates[['u10', 'v10']], df=[4, 4], degree=[3, 3])
    gam_bs = GLMGam.from_formula('u10_hr ~ u10 + tpi_500 + hour + aspect + blh + fsr + sp + z + tpi_500:hour',
                                 data=data, smoother=bs)
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    for col in ['hour', 'tpi_500', 'aspect', 'blh', 'fsr', 'sp', 'z']:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))
        d = data.groupby(col).agg({'u10':'mean', 'u10_hr':'mean',
                                   'v10':'mean', 'v10_hr':'mean',
                                   'wind_speed':lambda x: np.mean(np.log(x)),
                                   'wind_speed_hr':lambda x: np.mean(np.log(x))})
        for pred, ax in zip(['u10', 'v10', 'wind_speed'], [ax1, ax2, ax3]):
            ax.scatter(d.index, np.abs(d[f'{pred}_hr']-d[pred]), s=8, color='navy', label = f'{pred.upper()} Diff', alpha=0.5)
            ax.set_xlabel(col)
            ax.legend()
        fig.savefig(f'/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/plots/mean_dist_fn_of_{col}.png')



