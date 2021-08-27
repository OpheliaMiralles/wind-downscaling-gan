from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model


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
        # data = tf.keras.engine.data_adapter.expand_1d(data)
        low_res, high_res, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(low_res)[0]
        # Run forward pass on the discriminator
        noise = self.noise_generator(batch_size)
        fake_high_res = self.generator([low_res, noise], training=False)

        # Combine true (1) and fake (0) outputs
        # combined_high_res = tf.concat([high_res, fake_high_res], axis=0)
        # combined_low_res = tf.concat([low_res, low_res], axis=0)
        # combined_labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Add a small noise to the labels
        # combined_labels += 0.01 * tf.random.uniform(tf.shape(combined_labels))

        gamma = 10
        combined_high_res = (high_res + fake_high_res) / 2

        with tf.GradientTape() as reg_tape:
            reg_tape.watch(combined_high_res)
            out = self.discriminator([low_res, combined_high_res], training=False)
            # value_reg = 10 * tf.square(tf.reduce_sum(out))
        gradients = reg_tape.gradient(out, combined_high_res)
        # gradients = reg_tape.gradient(out, self.discriminator.trainable_weights)
        gradients = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
        # gradients = tf.concat([tf.reshape(g, (-1,)) for g in gradients], axis=0)
        gradient_reg = gamma * tf.reduce_mean((gradients - 1) ** 2)

        with tf.GradientTape() as disc_tape:
            true_predictions = self.discriminator([low_res, high_res], training=True)
            fake_predictions = self.discriminator([low_res, fake_high_res], training=True)
            disc_loss = self.discriminator.compiled_loss(true_predictions, fake_predictions, sample_weight,
                                                         regularization_losses=[gradient_reg])
            # predictions = self.discriminator([combined_low_res, combined_high_res], training=True)
            # disc_loss = self.discriminator.compiled_loss(combined_labels, predictions, sample_weight,
            #                                              regularization_losses=self.losses)

        # Run discriminator backward pass
        grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        # self.discriminator.optimizer.minimize(disc_loss, self.discriminator.trainable_variables, tape=disc_tape)
        # self.discriminator.compiled_metrics.update_state(combined_labels, predictions, sample_weight)

        # Run forward pass on generator using the discriminator to evaluate loss
        noise = self.noise_generator(batch_size)

        # For the generator, images should be predicted as true (1)
        # labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gen_tape:
            fake_high_res = self.generator([low_res, noise], training=True)
            fake_predictions = self.discriminator([low_res, fake_high_res], training=False)
            gen_loss = self.discriminator.compiled_loss(true_predictions, fake_predictions, sample_weight,
                                                        # regularization_losses=self.losses
                                                        )
        # Run generator backward pass
        grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        # self.generator.optimizer.minimize(gen_loss, self.generator.trainable_variables, tape=gen_tape)
        # self.generator.compiled_metrics.update_state(labels, fake_predictions, sample_weight)

        # Collect metrics to return
        return_metrics = {'loss': (gen_loss + disc_loss) / 2, 'd_loss': disc_loss, 'g_loss': gen_loss}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(x)[0]
        noise = self.noise_generator(batch_size)
        generated = self.generator([x, noise], training=False)
        pred_labels = self.discriminator([x, generated], training=False)
        true_labels = tf.ones((batch_size, 1))
        loss = self.discriminator.compiled_loss(true_labels, pred_labels)

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
                generator_loss,
                discriminator_loss,
                **kwargs):
        super().compile(**kwargs)
        self.generator.compile(generator_optimizer, generator_loss)
        self.discriminator.compile(discriminator_optimizer, discriminator_loss)

    def save_weights(self, filepath, *args, **kwargs):
        self.generator.save_weights(Path(filepath) / 'generator', *args, **kwargs)
        self.discriminator.save_weights(Path(filepath) / 'discriminator', *args, **kwargs)

    def load_weights(self,
                     filepath,
                     *args, **kwargs):
        self.generator.load_weights(Path(filepath) / 'generator', *args, **kwargs)
        self.discriminator.load_weights(Path(filepath) / 'discriminator', *args, **kwargs)
