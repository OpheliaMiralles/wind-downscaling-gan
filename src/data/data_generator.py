import pathlib

import numpy as np
import pandas as pd
import xarray as xr


# COSMO data for Swizerland has resolution 1.1km, which is an image size of 1075*691
# we might consider daily sequences in hourly resolution (24 images per sequence) because of observed patterns
# adding the altitude in addition to longitude and latitude would produce vectors in 5 dimensions (time, lon, lat, (alt, value))

class BatchGenerator(object):
    def __init__(self, path_to_data, decoder,
                 start_date=None,
                 end_date=None,
                 sequence_length=6,
                 patch_length_pixel=30,
                 batch_size=16,
                 transform=True,
                 input_variables=('u10', 'v10', 'blh', 'fsr', 'sp',
                                  'z', 'vo', 'd',
                                  'tpi_500', 'ridge_index_norm'),
                 output_variables=('U_10M', 'V_10M')):
        self.insert_random_img_transforms = transform
        self.batch_size = batch_size
        self.decoder = decoder
        self.sequence_length = sequence_length
        self.patch_length_pixel = patch_length_pixel
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.dates = sorted([f.with_suffix('').name.split('_')[-1] for f in pathlib.Path(path_to_data).glob('x_*.nc')])
        if start_date is not None:
            self.dates = [d for d in self.dates if pd.to_datetime(d) >= start_date]
        if end_date is not None:
            self.dates = [d for d in self.dates if pd.to_datetime(d) <= end_date]
        self.data_path = path_to_data
        self.current_date_index = -1
        self.prng = np.random.RandomState(seed=None)

    def next_date(self):
        self.current_date_index = (self.current_date_index + 1) % len(self.dates)
        return self.dates[self.current_date_index]

    def __iter__(self):
        return self

    def reset(self, random_seed=None):
        self.prng = np.random.RandomState(seed=random_seed)
        self.current_date_index = -1

    def get_random_square_sequences_per_day(self, X, Y):
        coords_keys = list(X.dims.keys())
        x_coord, y_coord = (k for k in coords_keys if k != 'time')
        random_x_coord = np.random.randint(0, X.dims[x_coord] - 1 - self.patch_length_pixel)
        random_y_coord = np.random.randint(0, X.dims[y_coord] - 1 - self.patch_length_pixel)
        random_time = np.random.randint(0, X.dims['time'] - 1 - self.sequence_length)
        random_x_sequence_on_the_upper_left_side = np.array(X.isel(
            {'time': slice(random_time, random_time + self.sequence_length),
             x_coord: slice(random_x_coord, random_x_coord + self.patch_length_pixel),
             y_coord: slice(random_y_coord, random_y_coord + self.patch_length_pixel)}).to_dask_dataframe()[
                                                                list(self.input_variables)]) \
            .reshape((self.sequence_length, self.patch_length_pixel, self.patch_length_pixel, -1))
        random_y_sequence_on_the_upper_left_side = np.array(Y.isel(
            {'time': slice(random_time, random_time + self.sequence_length),
             x_coord: slice(random_x_coord, random_x_coord + self.patch_length_pixel),
             y_coord: slice(random_y_coord, random_y_coord + self.patch_length_pixel)}).to_dask_dataframe()[
                                                                list(self.output_variables)]) \
            .reshape((self.sequence_length, self.patch_length_pixel, self.patch_length_pixel, -1))
        return random_x_sequence_on_the_upper_left_side, random_y_sequence_on_the_upper_left_side

    def __next__(self):
        date = self.next_date()
        input = xr.open_dataset(pathlib.Path(self.data_path, f'x_{date}').with_suffix('.nc'))
        output = xr.open_dataset(pathlib.Path(self.data_path, f'y_{date}').with_suffix('.nc'))
        input_batch = []
        output_batch = []
        for b in range(self.batch_size):
            X, Y = self.get_random_square_sequences_per_day(input, output)
            X = self.decoder(X)
            # Y = self.decoder(Y)
            if self.insert_random_img_transforms:
                X, Y = self.transform_sequence(X, Y)
            input_batch.append(X)
            output_batch.append(Y)
        in_batch = np.stack(input_batch, axis=0)
        out_batch = np.stack(output_batch, axis=0)
        return (in_batch, out_batch)

    def transform_sequence(self, X, Y):
        # mirror
        if bool(self.prng.randint(2)):
            X = np.flip(X, axis=1)
            Y = np.flip(Y, axis=1)
        if bool(self.prng.randint(2)):
            X = np.flip(X, axis=2)
            Y = np.flip(Y, axis=2)
        # rotate
        num_rot = self.prng.randint(4)
        if num_rot > 0:
            X = np.rot90(X, k=num_rot, axes=(1, 2))
            Y = np.rot90(Y, k=num_rot, axes=(1, 2))
        return X, Y

    def __call__(self):
        return next(self)


class NoiseGenerator(object):
    def __init__(self, noise_shape, std=10, random_seed=None):
        self.noise_shape = noise_shape
        self.prng = np.random.RandomState(seed=random_seed)
        self.std = std

    def __call__(self):
        mean = 0
        std = self.std
        bs = self.noise_shape[0]
        t = self.noise_shape[1]
        x = self.noise_shape[2]
        y = self.noise_shape[3]
        time_varying_noise = np.array([np.full((x, y), v) for v in self.prng.normal(mean, std, size=bs * t)]).reshape(
            (bs, t, x, y))
        lon_varying_noise = np.array([np.full((t, y), v) for v in self.prng.normal(mean, std, size=bs * x)]).reshape(
            (bs, t, x, y))
        lat_varying_noise = np.array([np.full((t, x), v) for v in self.prng.normal(mean, std, size=bs * y)]).reshape(
            (bs, t, x, y))
        lon_lat_noise = np.array([np.full(t, v) for v in self.prng.normal(mean, std, size=bs * x * y)]).reshape(
            (bs, t, x, y))
        noise = np.stack([time_varying_noise, lon_varying_noise, lat_varying_noise, lon_lat_noise]).reshape(
            self.noise_shape)
        return noise

    # def __call__(self, **kwds):
    #     num = np.arange(self.noise_shapes[4])
    #     def random_field(**kwds):
    #         bs = np.arange(self.noise_shapes[0])
    #         t = np.arange(self.noise_shapes[1])
    #         x = np.arange(self.noise_shapes[2])
    #         y = np.arange(self.noise_shapes[3])
    #         model = gs.Gaussian(dim=4, **kwds)
    #         srf = gs.SRF(model)#, generator='VectorField')
    #         field = srf((bs, t, x, y), mesh_type='structured')
    #         return field
    #     return np.array([random_field(**kwds) for _ in num]).reshape(self.noise_shapes)


class NaiveDecoder(object):
    def __init__(self, normalize=True):
        self.normalize_input = normalize

    def __call__(self, img):
        valid = (img != np.nan)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = img[valid]
        if self.normalize_input:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img - np.nanmean(img, axis=(0, 1, 2), keepdims=True)) / np.nanstd(img, axis=(0, 1, 2),
                                                                                  keepdims=True)

    def normalize_positive(self, img):
        return (img - np.nanmin(img, axis=(0, 1, 2), keepdims=True)) / (
                np.nanmax(img, axis=(0, 1, 2), keepdims=True) - np.nanmin(img, axis=(0, 1, 2), keepdims=True))

    def denormalize(self, img):
        img = img * np.nanstd(img) + np.nanmean(img)
        return img

    def denormalize_positive(self, img):
        return np.nanmin(img) + img * (np.nanmax(img) - np.nanmin(img))


class WindSpeedDecoder(object):
    def __init__(self, value_range=(np.log10(0.1), np.log10(100)),
                 below_val=np.nan, normalize=False):
        self.value_range = value_range
        self.below_val = below_val
        self.normalize_output = normalize

    def __call__(self, img):
        valid = (img != 0)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = img[valid]
        img_dec[img_dec < self.value_range[0]] = self.below_val
        img_dec.clip(max=self.value_range[1], out=img_dec)
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img - self.below_val) / \
               (self.value_range[1] - self.below_val)

    def denormalize(self, img, set_nan=True):
        img = img * (self.value_range[1] - self.below_val) + self.below_val
        img[img < self.value_range[0]] = self.below_val
        if set_nan:
            img[img == self.below_val] = np.nan
        return img


class WindComponentDecoder(object):
    def __init__(self, value_range=(-10, 10),
                 below_val=np.nan, normalize=True):
        self.value_range = value_range
        self.below_val = below_val
        self.normalize_output = normalize

    def __call__(self, img):
        valid = (img != 0)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = img[valid]
        img_dec[img_dec < self.value_range[0]] = self.below_val
        img_dec.clip(max=self.value_range[1], out=img_dec)
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img - np.mean(img)) / np.std(img)

    def denormalize(self, img, set_nan=True):
        img = img * np.std(img) + np.mean(img)
        img[img < self.value_range[0]] = self.below_val
        if set_nan:
            img[img == self.below_val] = np.nan
        return img
