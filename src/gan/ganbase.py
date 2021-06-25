import time
from itertools import islice
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import callbacks

from data.data_generator import NoiseGenerator


class GANCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, save_dir, filepath, model, *args, **kwargs):
        super(GANCheckpoint, self).__init__(filepath, *args, **kwargs)
        self.gan = model
        self.save_dir = Path(save_dir)

    def _save_model(self, epoch, logs):
        # save both the generator and the discriminator
        outfile = str(self.save_dir / 'generator' / self.filepath.format(epoch=epoch+1))
        self.gan.generator.save(outfile, overwrite=True, options=self._options)
        outfile = str(self.save_dir / 'discriminator' / self.filepath.format(epoch=epoch + 1))
        self.gan.discriminator.save(outfile, overwrite=True, options=self._options)


# This annotation causes the function to be "compiled".
# @tf.function
def train_step(gan, training_inputs, noise_init, noise_update, train_disc=True):
        inputs, outputs = training_inputs

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = gan.generator([inputs, noise_init[:,0,...], noise_update], training=True)

            real_output = gan.discriminator([inputs, outputs], training=train_disc)
            fake_output = gan.discriminator([inputs, generated_images], training=train_disc)

            gen_loss = gan.generator.loss(fake_output)
            disc_loss = gan.discriminator.loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
        gan.generator.optimizer.apply_gradients(zip(gradients_of_generator, gan.generator.trainable_variables))

        if train_disc:
            gradients_of_discriminator = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)
            gan.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

        return {'generator_loss': gen_loss, 'discriminator_loss': disc_loss}


class GAN(Model):
    def __init__(self, generator: Model, discriminator: Model,
                 noise_dim, initial_state, gen_disc_ratio: int = 1):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.initial_state = initial_state
        checkpoint = GANCheckpoint('./checkpoints', 'weights-{epoch:02d}.hdf5', model=self, verbose=1)
        board = callbacks.TensorBoard(update_freq=1, profile_batch=(2, 100))
        self.callback_list = callbacks.CallbackList([checkpoint, board], add_history=True, add_progbar=True, model=self)
        self.gen_disc_ratio = gen_disc_ratio

    def train(self, dataset, epochs, batches_per_epoch=None, verbose=False):
        self.callback_list.set_params(dict(verbose=verbose, epochs=epochs, steps=batches_per_epoch))
        self.callback_list.on_train_begin()
        batch_per_epoch = batches_per_epoch or dataset.input_sequences.shape[0] // dataset.batch_size
        for epoch in range(epochs):
            start = time.time()
            self.callback_list.on_epoch_begin(epoch)

            for i, image_batch in enumerate(islice(dataset, batch_per_epoch)):
                self.callback_list.on_train_batch_begin(i)
                batch_start = time.time()
                inputs, outputs = image_batch
                batch_size, t, x, y = inputs.shape[:-1]
                noise_update = NoiseGenerator((batch_size, t, x, y, self.noise_dim))()
                noise_init = NoiseGenerator((batch_size, 1, x, y, self.noise_dim))()
                losses = train_step(self, image_batch, noise_init, noise_update, train_disc=i % self.gen_disc_ratio == 0)
                self.callback_list.on_train_batch_end(i, logs=losses)
                if i >= batch_per_epoch:
                    break

            self.callback_list.on_epoch_end(epoch)
        self.callback_list.on_train_end()

    def build(self, input_shape):
        out_shape = self.generator.build(input_shape)
        self.discriminator.build([input_shape, out_shape])

    def compile(self,
                generator_optimizer,
                discriminator_optimizer,
                generator_loss,
                discriminator_loss,
                **kwargs):
        super().compile(**kwargs)
        self.generator.compile(generator_optimizer, generator_loss)
        self.discriminator.compile(discriminator_optimizer, discriminator_loss)
