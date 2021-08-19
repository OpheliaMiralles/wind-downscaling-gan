from tensorflow_gan import gan_model, gan_train, gan_loss, gan_train_ops
from tensorflow_gan.python.estimator import GANEstimator

from data.data_generator import BatchGenerator, NaiveDecoder
from models import make_generator_model, make_discriminator_model
from train import retrieve_data, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer


def make_gan():
    x_train, y_train = retrieve_data()
    gan = gan_model(make_generator_model, make_discriminator_model, y_train, x_train)
    loss = gan_loss(gan, generator_loss, discriminator_loss)
    train_ops = gan_train_ops(gan, loss, generator_optimizer(), discriminator_optimizer())
    gan_train(train_ops, './logs')
    return gan


def make_estimator():
    gan_estimator = GANEstimator('./estimator', make_generator_model, make_discriminator_model, generator_loss,
                                 discriminator_loss, generator_optimizer(), discriminator_optimizer())
    x_train, y_train = retrieve_data(10)
    data = BatchGenerator(x_train, y_train, NaiveDecoder())
    gan_estimator.train(data, steps=1)
    return gan_estimator
