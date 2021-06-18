import os

import numpy as np
import pandas as pd
import xarray as xr


class PointPredictionProcessing(object):
    def __init__(self, path_to_input_surface: str,
                 path_to_input_z500: str,
                 path_to_static_input: str,
                 output: pd.DataFrame,
                 patch_size: int,
                 variables_to_predict=('u10', 'v10'),
                 static_predictors=('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                                    'we_derivative', 'sn_derivative',
                                    'slope', 'aspect'),
                 predictors_surface=('blh', 'fsr', 'sp', 'sshf',
                                     'd2m', 't2m',
                                     'u100', 'v100', 'u10', 'v10'),
                 predictors_z500=('d', 'z', 'u', 'v', 'w', 'vo'),
                 start_period=pd.to_datetime('2016-01-10'),
                 end_period=pd.to_datetime('2020-12-31'),
                 num_days=None):
        """

        :param path_to_input: input files, likely images in low resolution (.nc individual files). the path can
        ether be the exact path to directory where the inputs are stored, or a pattern to individual files.
        :param path_to_output: output, likely csv containing values for specific stations
        :param patch_size: for point prediction, we need to build a patch around the point we aim at predicting.
                the parameter patches_size sets the size of the neighborhood (in pixel) around points that will be the low-res reference.
        :param num_days: number days to process, None for all
        """
        self.input_surface = xr.open_mfdataset(path_to_input_surface)
        self.input_z500 = xr.open_mfdataset(path_to_input_z500)
        self.static_input = xr.open_mfdataset(path_to_static_input)
        self.output = output
        self.output.date = pd.to_datetime(self.output.date)
        self.output = self.output[(self.output.date >= start_period) & (self.output.date < end_period)]
        self.patch_size = patch_size
        self.predictors_surface = predictors_surface
        self.predictors_z500 = predictors_z500
        self.static_predictors = static_predictors
        self.variables_to_predict = variables_to_predict
        self.num_days = num_days

    def create_patch_for_point(self, date, lon_point, lat_point):
        date = date.strftime("%Y-%m-%d")

        def inputs_rectangle_around_point(lon_point=lon_point, lat_point=lat_point):
            inputs_to_concatenate = []
            for df, predictors in zip([self.input_surface, self.input_z500],
                                      [self.predictors_surface, self.predictors_z500]):
                distances_lon, distances_lat = np.abs(np.array(df.longitude.data) - lon_point), np.abs(
                    np.array(df.latitude.data) - lat_point)
                nearest_lon, nearest_lat = np.argmin(distances_lon), \
                                           np.argmin(distances_lat)
                lon1, lon2 = nearest_lon - self.patch_size // 2, nearest_lon + self.patch_size // 2
                lat1, lat2 = nearest_lat - self.patch_size // 2, nearest_lat + self.patch_size // 2
                inputs = df.sel(time=date).isel(longitude=slice(lon1, lon2 + 1), latitude=slice(lat1, lat2 + 1))
                inputs = inputs.drop_vars([v for v in inputs.variables if v not in predictors])
                new_dim = (inputs.dims['time'], inputs.dims['longitude'], inputs.dims['latitude'], len(inputs))
                inputs = inputs.to_dataframe()[list(predictors)].to_numpy().reshape(new_dim)
                inputs_to_concatenate.append(inputs)
            return np.concatenate(inputs_to_concatenate, axis=3)

        inputs = inputs_rectangle_around_point()

        def static_inputs_rectangle_around_point(time_steps=inputs.shape[0], lon_point=lon_point, lat_point=lat_point):
            distances_lon, distances_lat = np.abs(np.array(self.static_input.x.data) - lon_point), np.abs(
                np.array(self.static_input.y.data) - lat_point)
            nearest_lon, nearest_lat = np.argmin(distances_lon), \
                                       np.argmin(distances_lat)
            lon1, lon2 = nearest_lon - self.patch_size // 2, nearest_lon + self.patch_size // 2
            lat1, lat2 = nearest_lat - self.patch_size // 2, nearest_lat + self.patch_size // 2
            static_inputs = self.static_input.isel(x=slice(lon1, lon2 + 1),
                                                   y=slice(lat1, lat2 + 1))
            static_inputs = static_inputs.drop_vars(
                [v for v in static_inputs.variables if v not in self.static_predictors])
            new_dim_static = (static_inputs.dims['x'], static_inputs.dims['y'], len(static_inputs))
            static_inputs = static_inputs.to_dataframe()[list(self.static_predictors)].to_numpy().reshape(
                new_dim_static)
            static_inputs = np.repeat(static_inputs[np.newaxis, ...], time_steps, axis=0)
            return static_inputs

        static_inputs = static_inputs_rectangle_around_point()

        dims_i1 = inputs.shape[:-1]
        dims_i2 = static_inputs.shape[:-1]
        if dims_i1 == dims_i2:
            return np.concatenate([inputs, static_inputs], axis=3)
        else:
            return None

    def get_input_and_output_arrays(self):
        inputs = []
        outputs = []
        lon_column = [c for c in self.output.columns if c.startswith('lon')][0]
        lat_column = [c for c in self.output.columns if c.startswith('lat')][0]
        station_column = [c for c in self.output.columns if c.startswith('station')][0]
        if 'date' not in self.output.columns:
            raise ValueError('date column not found in output, cannot synchronize data.')
        input_dates = np.intersect1d(pd.to_datetime(self.input_surface.time.data).normalize().unique().sort_values(),
                                     pd.to_datetime(self.input_z500.time.data).normalize().unique().sort_values())
        if self.num_days is not None:
            input_dates = input_dates[:self.num_days]
        # one file per day
        for date in pd.DatetimeIndex(self.output.date).unique().sort_values():
            # checks if data in input
            if date in input_dates:
                print(f"Processing {date}")
                num_stations = 0
                output_for_date = self.output[self.output.date == date]
                output_for_date['datetime'] = pd.to_datetime(output_for_date['datetime'])
                output_for_date = output_for_date.sort_values('datetime')
                complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
                output_for_date = output_for_date[
                    output_for_date[station_column].isin(complete_stations[complete_stations].index)]
                for s in sorted(output_for_date[station_column].unique()):
                    output = output_for_date[output_for_date[station_column] == s]
                    longitude = float(output[lon_column].unique())
                    latitude = float(output[lat_column].unique())
                    new_output = np.asarray(output[list(self.variables_to_predict)]).reshape(len(output), 1, 1, len(
                        self.variables_to_predict))
                    hours, lons, lats, _ = new_output.shape
                    if hours != 24:
                        continue
                    new_input = self.create_patch_for_point(date, longitude, latitude)
                    if new_input is None:
                        continue
                    _, lons, lats, _ = new_input.shape
                    if lons != self.patch_size or lats != self.patch_size:
                        continue
                    inputs.append(new_input)
                    outputs.append(new_output)
                    num_stations += 1
                print(f"Processed {num_stations} stations")
            else:
                print(f'Date {date} has no complete match in inputs, cannot synchronize data.')
        return np.asarray(inputs), np.asarray(outputs)

    def get_input_output_dataframe_from_array(self, path_to_arrays, aggregation_method='nearest'):
        obs = self.output
        obs['date'] = pd.to_datetime(obs['date'])
        start_train_period = pd.to_datetime('2016-01-10')
        end_train_period = pd.to_datetime('2016-06-10')
        data = []
        for d in pd.date_range(start_train_period, end_train_period):
            d_str = d.strftime('%Y%m%d')
            x = f'{path_to_arrays}/x_train_{d_str}.npy'
            y = f'{path_to_arrays}/y_train_{d_str}.npy'
            if aggregation_method == 'nearest':
                xarr = np.load(x)[:, :, 1, 1, ...]
            elif aggregation_method == 'mean':
                xarr = np.mean(np.load(x), axis=(2, 3))
            else:
                print('Defaulting aggregation to nearest neighbor...')
                xarr = np.load(x)[:, :, 1, 1, ...]
            m, n, r = xarr.shape
            xarr = np.column_stack((np.repeat(np.arange(m), n), xarr.reshape(m * n, -1)))
            yarr = np.load(y)[:, :, 0, 0, ...]
            m, n, r = yarr.shape
            yarr = np.column_stack((np.repeat(np.arange(m), n), yarr.reshape(m * n, -1)))
            interest_var = tuple(f'{v}_hr' for v in self.variables_to_predict)
            out_df = pd.DataFrame(np.concatenate([xarr, yarr[:, 1:]], axis=-1), columns=list(tuple('station') +
                                                                                             self.predictors_surface +
                                                                                             self.predictors_z500 +
                                                                                             self.static_predictors +
                                                                                             interest_var))
            out_df['hour'] = np.repeat(np.arange(n), m)
            out_df['date'] = d
            out_df = out_df.assign(datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H'))
            output_for_date = obs[obs['date'] == d]
            station_column = 'station'
            complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
            output_for_date = output_for_date[
                output_for_date[station_column].isin(complete_stations[complete_stations].index)]
            out_df['station'] = output_for_date.reset_index()[station_column]
            data.append(out_df)
        data = pd.concat(data)
        return data