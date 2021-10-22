import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model


class GAN(Model):
    def __init__(self, generator: Model, discriminator: Model, noise_generator, n_critic=2,
                 generator_additional_losses=None, *args,
                 **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator
        self._generator_additional_losses = generator_additional_losses
        self._steps_per_execution = tf.convert_to_tensor(1)
        self._n_critic = n_critic

    def _assert_compile_was_called(self):
        return self.generator._assert_compile_was_called() and self.discriminator._assert_compile_was_called()

    def train_step(self, data):
        gamma = 10
        low_res, high_res, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(low_res)[0]
        # Train discriminator during _n_critic steps before updating the generator
        for t in range(self._n_critic):
            # Compute gradient penalty
            noise = self.noise_generator(batch_size)
            fake_high_res = self.generator([low_res, noise], training=False)
            eps = tf.random.uniform(shape=(batch_size, 1, 1, 1, 1), minval=0, maxval=1)
            combined_high_res = eps * high_res + (1 - eps) * fake_high_res
            with tf.GradientTape() as reg_tape:
                reg_tape.watch(combined_high_res)
                out = self.discriminator([low_res, combined_high_res], training=False)
            gradients = reg_tape.gradient(out, combined_high_res)
            gradients = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
            gradient_reg = gamma * tf.reduce_mean((gradients - 1) ** 2)
            # Run forward pass on the discriminator
            with tf.GradientTape() as disc_tape:
                high_res_score = self.discriminator([low_res, high_res], training=True)
                fake_high_res_score = self.discriminator([low_res, fake_high_res], training=True)
                disc_loss = self.discriminator.compiled_loss(high_res_score, fake_high_res_score, sample_weight,
                                                             regularization_losses=[gradient_reg])
            grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Run forward pass on the generator
        with tf.GradientTape() as gen_tape:
            noise = self.noise_generator(batch_size)
            fake_high_res = self.generator([low_res, noise], training=True)
            fake_high_res_score = self.discriminator([low_res, fake_high_res], training=False)
            gen_disc_loss = -tf.reduce_mean(fake_high_res_score)  # disc score for fake outputs
            if self._generator_additional_losses is not None:
                gen_additional_losses = tf.reduce_mean(
                    [loss(high_res, fake_high_res) for loss in self._generator_additional_losses])
                gen_loss = (gen_disc_loss + gen_additional_losses) / 2
            else:
                gen_loss = gen_disc_loss
        grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.generator.compiled_metrics.update_state(high_res, fake_high_res, sample_weight)
        self.compiled_metrics.update_state(high_res_score, fake_high_res_score, sample_weight)
        disc_loss_unregularized = self.discriminator.compiled_loss(high_res_score, fake_high_res_score, sample_weight)

        # Collect metrics to return
        return_metrics = {'d_loss': disc_loss_unregularized,
                          'g_loss': gen_loss}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        for metric in self.generator.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[f'g_{metric.name}'] = result
        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(x)[0]
        noise = self.noise_generator(batch_size)
        true_predictions = self.discriminator([x, y])
        generated = self.generator([x, noise], training=False)
        fake_predictions = self.discriminator([x, generated], training=False)
        loss = self.discriminator.compiled_loss(true_predictions, fake_predictions)

        # Collect metrics to return
        return_metrics = {'loss': loss}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def compile(self,
                generator_optimizer,
                discriminator_optimizer,
                generator_loss=None,
                generator_metrics=None,
                discriminator_loss=None,
                **kwargs):
        super().compile(**kwargs)
        self.generator.compile(generator_optimizer, generator_loss, metrics=generator_metrics)
        self.discriminator.compile(discriminator_optimizer, discriminator_loss)

    def call(self, inputs, training=None, mask=None):
        low_res, high_res, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        batch_size = tf.shape(low_res)[0]
        noise = self.noise_generator(batch_size)
        return self.generator.call([low_res, noise], training=training, mask=mask)

    def save_weights(self, filepath, *args, **kwargs):
        self.generator.save_weights(os.path.join(filepath, 'generator'), *args, **kwargs)
        self.discriminator.save_weights(os.path.join(filepath, 'discriminator'), *args, **kwargs)

    def load_weights(self,
                     filepath,
                     *args, **kwargs):
        self.generator.load_weights(Path(filepath) / f'generator', *args, **kwargs)
        self.discriminator.load_weights(Path(filepath) / f'discriminator', *args, **kwargs)
