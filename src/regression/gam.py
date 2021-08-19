import pathlib

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

from notebooks.visualisation_utils import HigherResPlateCarree


def normalizer(data):
    return (data - np.nanmean(data)) / np.nanstd(data)


def positive_normalizer(data):
    return (data - data.min()) / (data.max() - data.min())


if __name__ == '__main__':
    data_path = pathlib.Path('')
    data = pd.read_csv(data_path)
    data = data.dropna(subset=['u10', 'v10'])
    data = data.assign(wind_speed=lambda x: np.sqrt(x['U_10M'] ** 2 + x['V_10M'] ** 2)).assign(
        wind_speed_hr=lambda x: np.sqrt(x['u10'] ** 2 + x['v10'] ** 2))
    data['u10'] = -data['u10']
    to_concat = []
    for s, d in data.groupby('station'):
        d = d[(d['wind_speed_hr'] > 0) & (d['wind_speed'] > 0)]
        u10_hr = -d['u10']
        ws_hr = np.log(d['wind_speed_hr'])
        d = d.assign(log_wind_speed=lambda x: np.log(x['wind_speed']))
        covariates = d[['U_10M_mean', 'U_10M', 'hour']]  # 'tpi_500', 'aspect', sp, hour, blh
        normalized_covariates = covariates.copy()
        for col in covariates:
            if col in ['tpi_500', 'ridge_index_norm',
                       'ridge_index_dir', 'we_derivative', 'sn_derivative', 'slope', 'aspect', 'altitude_m', 'lon',
                       'lat']:
                pass
            elif np.min(covariates[col]) >= 0 and col != 'hour':
                normalized_covariates[col] = positive_normalizer(normalized_covariates[col])
            else:
                normalized_covariates[col] = normalizer(normalized_covariates[col])
        # normalized_covariates['intercept'] = 1.
        mod = sm.OLS(normalizer(u10_hr), normalized_covariates)
        res = mod.fit()
        to_concat.append(
            pd.DataFrame([[s, d['altitude_m'].unique()[0], d['lon'].unique()[0], d['lat'].unique()[0],
                           d['tpi_500'].unique()[0], d['ridge_index_norm'].unique()[0],
                           d['ridge_index_dir'].unique()[0],
                           d['we_derivative'].unique()[0], d['sn_derivative'].unique()[0],
                           d['slope'].unique()[0],
                           res.rsquared, res.rsquared_adj, res.aic, res.bic]],
                         columns=['station', 'altitude_m', 'lon', 'lat',
                                  'tpi_500', 'ridge_index_norm',
                                  'ridge_index_dir', 'we_derivative', 'sn_derivative', 'slope',
                                  'Rsquared', 'Rsquared_adj', 'AIC', 'BIC']))
    res_ols = pd.concat(to_concat)
    res_ols = res_ols.sort_values('altitude_m').set_index('altitude_m')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    for col, color in zip([c for c in ['Rsquared', 'Rsquared_adj', 'AIC', 'BIC']],
                          ['navy', 'royalblue', 'salmon', 'chocolate']):
        if col in ['AIC', 'BIC']:
            ax = ax2
        else:
            ax = ax1
        ax.scatter(res_ols.index, res_ols[col],
                   s=8, color=color, marker='x', label=col)
        ax.set_xlabel('station altitude (m)')
    ax1.legend()
    ax2.legend()
    fig.suptitle('OLS regression results per station')


    def plot_stations(ax, df):
        ax.set_extent(
            [df.lon.min() - 0.3, df.lon.max() + 0.3, df.lat.min() - 0.3, df.lat.max() + 0.3])
        ax.coastlines()
        # ax.stock_img()
        ax.add_feature(cartopy.feature.LAND, color='goldenrod')
        ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color='royalblue')
        ax.add_feature(cartopy.feature.OCEAN, color='skyblue')
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='grey')
        c_scheme = ax.scatter(x=df.lon, y=df.lat,
                              s=150,
                              c=df.Rsquared,
                              cmap='RdYlGn',
                              alpha=0.5,
                              transform=HigherResPlateCarree())
        plt.colorbar(c_scheme, location='bottom', pad=0.07,
                     label='R Squared', ax=ax)
        for s, d in df.groupby('station'):
            ax.text(d.lon + 0.1, d.lat - 0.04, s, transform=HigherResPlateCarree(),
                    fontsize=10, c='black', horizontalalignment='center', verticalalignment='center')
        return ax


    subplot_kw = {'projection': HigherResPlateCarree()}
    fig, ax1 = plt.subplots(ncols=1, subplot_kw=subplot_kw, figsize=(17, 12.5))
    plot_stations(ax1, res_ols)

    bs = BSplines(data[['hour']], df=[4, 4], degree=[3, 3])
    gam_bs = GLMGam.from_formula('u10 ~ 1 + U_10M + U_10M_mean + slope + hour + aspect',
                                 data=data, smoother=bs)
    res_bs = gam_bs.fit()
    print(res_bs.summary())
    datafitted = data.assign(fitted=res_bs.fittedvalues)
    for s, d in datafitted.groupby('station'):
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
        perhour = d.groupby('hour').agg({'u10': 'mean', 'fitted': 'mean'})
        perhour.plot(ax=ax)
        fig.suptitle(s)
