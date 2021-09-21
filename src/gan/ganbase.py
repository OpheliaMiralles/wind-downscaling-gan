import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class GAN(Model):
    def __init__(self, generator: Model, discriminator: Model, noise_generator, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator
        self._steps_per_execution = tf.convert_to_tensor(1)

    def _assert_compile_was_called(self):
        return self.generator._assert_compile_was_called() and self.discriminator._assert_compile_was_called()

    def train_step(self, data):
        low_res, high_res, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(low_res)[0]
        # Run forward pass on the discriminator
        noise = self.noise_generator(batch_size)
        fake_high_res = self.generator([low_res, noise], training=False)
        gamma = 10
        eps = tf.random.uniform(shape=(batch_size, 1, 1, 1, 1), minval=0, maxval=1)
        combined_high_res = eps * high_res + (1 - eps) * fake_high_res

        with tf.GradientTape() as reg_tape:
            reg_tape.watch(combined_high_res)
            out = self.discriminator([low_res, combined_high_res], training=False)
        gradients = reg_tape.gradient(out, combined_high_res)
        gradients = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
        gradient_reg = gamma * tf.reduce_mean((gradients - 1) ** 2)

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            fake_high_res = self.generator([low_res, noise], training=True)
            high_res_score = self.discriminator([low_res, high_res], training=True)
            fake_high_res_score = self.discriminator([low_res, fake_high_res], training=True)
            disc_loss = self.discriminator.compiled_loss(high_res_score, fake_high_res_score, sample_weight)
                                                         #regularization_losses=[gradient_reg])
            gen_loss = tf.reduce_mean(fake_high_res_score)  # disc score for fake outputs
        grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        # Update metrics
        self.generator.compiled_metrics.update_state(high_res, fake_high_res, sample_weight)
        self.compiled_metrics.update_state(high_res_score, fake_high_res_score, sample_weight)
        disc_loss_unregularized = self.discriminator.compiled_loss(high_res_score, fake_high_res_score, sample_weight)
        grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

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

    def save_weights(self, filepath, *args, **kwargs):
        self.generator.save_weights(os.path.join(filepath, 'generator'), *args, **kwargs)
        self.discriminator.save_weights(os.path.join(filepath, 'discriminator'), *args, **kwargs)

    def load_weights(self,
                     filepath,
                     *args, **kwargs):
        self.generator.load_weights(Path(filepath) / f'generator', *args, **kwargs)
        self.discriminator.load_weights(Path(filepath) / f'discriminator', *args, **kwargs)
