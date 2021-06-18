import numpy as np


# COSMO data for Swizerland has resolution 1.1km, which is an image size of 1075*691
# we might consider daily sequences in hourly resolution (24 images per sequence) because of observed patterns
# adding the altitude in addition to longitude and latitude would produce vectors in 5 dimensions (time, lon, lat, (alt, value))

class BatchGenerator(object):
    def __init__(self, input_sequences, output_sequences, decoder, batch_size=32,
                 transform=True):
        self.insert_random_img_transforms = transform
        self.batch_size = batch_size
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.decoder = decoder
        self.reset()

    def __iter__(self):
        return self

    def reset(self, random_seed=None):
        self.prng = np.random.RandomState(seed=random_seed)
        self.next_ind = np.array([], dtype=int)

    def next_indices(self):
        N = self.input_sequences.shape[0]  # nb of sequences
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(N, dtype=int)
            self.prng.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])
        return self.next_ind[:self.batch_size]

    def __next__(self):
        ind = self.next_indices()
        self.next_ind = self.next_ind[self.batch_size:]
        X = self.input_sequences[ind, ...]
        X = self.decoder(X)
        Y = self.output_sequences[ind, ...]
        Y = self.decoder(Y)
        if self.insert_random_img_transforms:
            X, Y = self.augment_sequence_batch(X, Y)
        X = self.decoder.normalize(X)
        Y = self.decoder.normalize(Y)
        return (X, Y)

    def tansform_sequence(self, X, Y):
        X = X.copy()
        Y = Y.copy()
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

    def augment_sequence_batch(self, X, Y):
        X = X.copy()
        Y = Y.copy()
        for i in range(X.shape[0]):
            X[i, ...], Y[i, ...] = self.tansform_sequence(X[i, ...], Y[i, ...])
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
    def __init__(self, normalize=False):
        self.normalize_output = normalize

    def __call__(self, img):
        valid = (img != np.nan)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = img[valid]
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img - np.nanmean(img)) / np.nanstd(img)

    def denormalize(self, img):
        img = img * np.nanstd(img) + np.nanmean(img)
        return img


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
