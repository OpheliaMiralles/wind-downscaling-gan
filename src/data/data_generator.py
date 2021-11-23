import abc
import os
import subprocess
import tempfile
from contextlib import contextmanager

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import parse
import tensorflow as tf
import xarray as xr


class Provider(abc.ABC):
    available_dates: Collection[str]

    @abc.abstractmethod
    def load(self, date: str) -> os.PathLike:
        ...

    def unload(self, loaded: os.PathLike) -> None:
        pass

    @contextmanager
    def provide(self, date):
        loaded = None
        try:
            loaded = self.load(date)
            yield loaded
        finally:
            if loaded is not None:
                self.unload(loaded)


class LocalFileProvider(Provider):
    def __init__(self, path_to_data: os.PathLike, pattern: str):
        self.data_path = Path(path_to_data)
        if '{date' not in pattern:
            raise ValueError("Expected a {date:fmt} placeholder, got " + pattern)
        if '{date}' in pattern:
            # Ensure we only parse %Y%m%d and not additional text
            pattern = pattern.replace('{date}', '{date:d}')
        self.pattern = pattern

    @cached_property
    def available_dates(self):
        dates = set()
        for f in self.data_path.iterdir():
            res = parse.parse(self.pattern, str(f.relative_to(self.data_path)))
            if res is not None:
                dates.add(str(res['date']))
        return dates

    def load(self, date: str) -> os.PathLike:
        return self.data_path / self.pattern.format(date=int(date))


class S3FileProvider(Provider):
    def __init__(self, bucket: str, *subfolders: str, pattern: str = None):
        if pattern is None:
            pattern = subfolders[-1]
            subfolders = subfolders[:-1]
        self.bucket = '/'.join([bucket] + list(subfolders))
        if '{date}' not in pattern:
            raise ValueError("Expected a {date} placeholder, got " + pattern)
        self.pattern = pattern

    @cached_property
    def available_dates(self):
        result = subprocess.run(['s3cmd', 'ls', f's3://{self.bucket}/'], capture_output=True)
        dates = set()
        for line in result.stdout.decode().splitlines():
            res = parse.search(f's3://{self.bucket}/' + self.pattern, line)
            if res is not None:
                dates.add(str(res.named['date']))
        return dates

    def load(self, date: str) -> str:
        dest = tempfile.mkdtemp()
        # Download the file from S3
        subprocess.run(['s3cmd', 'get', f's3://{self.bucket}/{self.pattern.format(date=int(date))}', str(dest) + '/'])
        return f'{dest}/{self.pattern.format(date=int(date))}'

    def unload(self, loaded: os.PathLike) -> None:
        # Remove the downloaded file
        Path(loaded).unlink(missing_ok=True)


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 input_provider: Provider,
                 output_provider: Provider,
                 decoder,
                 start_date=None,
                 end_date=None,
                 sequence_length=6,
                 patch_length_pixel=30,
                 batch_size=16,
                 transform=True,
                 input_variables=('u10', 'v10', 'blh', 'fsr', 'sp',
                                  'z', 'vo', 'd',
                                  'tpi_500', 'ridge_index_norm'),
                 output_variables=('U_10M', 'V_10M'),
                 num_workers=1,
                 ):
        self.num_workers = num_workers
        self._bg = _BatchGenerator(input_provider, output_provider, decoder, start_date, end_date,
                                   sequence_length, patch_length_pixel,
                                   batch_size, transform, input_variables, output_variables)
        if self.num_workers > 1:
            self.enqueuer = tf.keras.utils.OrderedEnqueuer(self._bg, use_multiprocessing=True)
        else:
            self.enqueuer = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        timestamps = pd.to_datetime(self._bg.dates)
        num_days = (max(timestamps) - min(timestamps)).days + 1
        return num_days

    def __getitem__(self, item):
        date = self._bg.dates[item]
        return self._bg.generate(date)

    def __enter__(self):
        if self.enqueuer is None:
            return self._bg
        if self.enqueuer.is_running():
            raise RuntimeError("Batch generator is already running!")
        self.enqueuer.start(workers=self.num_workers, max_queue_size=8)
        return self.enqueuer.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enqueuer is not None:
            self.enqueuer.stop(60)


class _BatchGenerator(object):

    def __init__(self,
                 input_provider: Provider,
                 output_provider: Provider,
                 decoder,
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
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.input_provider = input_provider
        self.output_provider = output_provider
        dates = set(self.input_provider.available_dates).intersection(self.output_provider.available_dates)
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            dates = [d for d in dates if pd.to_datetime(d) >= start_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            dates = [d for d in dates if pd.to_datetime(d) <= end_date]
        self.dates = sorted(dates)
        self.reset()

    def next_date(self):
        self.current_date_index = (self.current_date_index + 1) % len(self.dates)
        return self.dates[self.current_date_index]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def reset(self, random_seed=None):
        self.prng = np.random.RandomState(seed=random_seed)
        self.current_date_index = -1

    def get_random_square_sequences_per_day(self, X, Y):
        x_coord, y_coord = 'x_1', 'y_1'
        random_x_coord = np.random.randint(0, X.dims[x_coord] - 1 - self.patch_length_pixel)
        random_y_coord = np.random.randint(0, X.dims[y_coord] - 1 - self.patch_length_pixel)
        random_time = np.random.randint(0, X.dims['time'] - 1 - self.sequence_length)

        def crop_to_array(x, variables):
            variables_and_coords = ['time', x_coord, y_coord] + variables
            patch = x.isel({
                'time': slice(random_time, random_time + self.sequence_length),
                x_coord: slice(random_x_coord, random_x_coord + self.patch_length_pixel),
                y_coord: slice(random_y_coord, random_y_coord + self.patch_length_pixel)
            })[variables_and_coords]
            if 'elevation' in variables_and_coords:
                patch["elevation"] = patch["elevation"] / 1e3
            to_stack = []
            for v in variables:
                to_stack.append(patch[v].to_numpy())
            return np.stack(to_stack, axis=-1)

        return crop_to_array(X, self.input_variables), crop_to_array(Y, self.output_variables)

    def generate(self, date):
        with self.input_provider.provide(date) as input_path, \
                self.output_provider.provide(date) as output_path:
            input = xr.open_dataset(input_path)
            output = xr.open_dataset(output_path)
            input_batch = []
            output_batch = []
            for b in range(self.batch_size):
                X, Y = self.get_random_square_sequences_per_day(input, output)
                X = self.decoder(X)
                if self.insert_random_img_transforms:
                    X, Y = self.transform_sequence(X, Y)
                input_batch.append(X)
                output_batch.append(Y)
            in_batch = np.stack(input_batch, axis=0)
            out_batch = np.stack(output_batch, axis=0)
            return (in_batch, out_batch)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, item):
        return self.generate(self.dates[item])

    def __next__(self):
        date = self.next_date()
        return self.generate(date)

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
        self.prng = (tf.random.Generator.from_seed(random_seed)
                     if random_seed is not None
                     else tf.random.get_global_generator())
        self.std = std

    def __call__(self, bs=None):
        mean = 0
        std = self.std
        bs = self.noise_shape[0] if bs is None else bs
        t = self.noise_shape[1]
        x = self.noise_shape[2]
        y = self.noise_shape[3]
        time_varying_noise = tf.reshape(tf.repeat(self.prng.normal((bs, t), mean, std), x * y), (bs, t, x, y))
        lon_varying_noise = tf.reshape(tf.repeat(self.prng.normal((bs, x), mean, std), t * y), (bs, t, x, y))
        lat_varying_noise = tf.reshape(tf.repeat(self.prng.normal((bs, y), mean, std), t * x), (bs, t, x, y))
        lon_lat_noise = tf.reshape(tf.repeat(self.prng.normal((bs, x, y), mean, std), t), (bs, t, x, y))
        noise = tf.stack([time_varying_noise, lon_varying_noise, lat_varying_noise, lon_lat_noise], axis=-1)
        return noise


class FlexibleNoiseGenerator(object):
    def __init__(self, noise_shape, std=1, random_seed=None):
        self.noise_shape = noise_shape
        self.prng = (tf.random.Generator.from_seed(random_seed)
                     if random_seed is not None
                     else tf.random.get_global_generator())
        self.std = std

    def __call__(self, bs=None, channels=None, std=None):
        mean = 0
        bs = self.noise_shape[0] if bs is None else bs
        t = self.noise_shape[1]
        x = self.noise_shape[2]
        y = self.noise_shape[3]
        channels = self.noise_shape[4] if channels is None else channels
        std = std or self.std
        return self.prng.normal(tf.convert_to_tensor((bs, t, x, y, channels), dtype=tf.int32), mean=mean, stddev=std)


class NaiveDecoder(object):
    def __init__(self, normalize=True):
        self.normalize_input = normalize

    def __call__(self, img):
        if self.normalize_input:
            img = self.normalize(img)
        return img

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
