import tensorflow as tf
import tensorlayer as tl

def build_encoder(origin_units, hidden_units, latent_units):
    init = tf.initializers.he_uniform()
    ni = Input((None, origin_units))
    nn = Dense(hidden_units, act = tf.nn.relu, W_init = init, b_init = init)(ni)
    mean = Dense(latent_units, W_init = init, b_init = init)(nn)
    logvar = Dense(latent_units, W_init = init, b_init = init)(nn)
    return tl.models.Model(inputs = ni, outputs = [mean, logvar])

def build_decoder(origin_units, hidden_units, latent_units):
    init = tf.initializers.he_uniform()
    ni = Input((None, latent_units))
    nn = Dense(hidden_units, act = tf.nn.relu, W_init = init, b_init = init)(ni)
    no = Dense(origin_units, act = tf.nn.sigmoid, W_init = init, b_init = init)(nn)
    return tl.models.Model(inputs = ni, outputs = no)

def build(origin_units, hidden_units, latent_units):
    encoder = build_encoder(origin_units, hidden_units, latent_units)
    decoder = build_decoder(origin_units, hidden_units, latent_units)
    def build_train_model():
        ni = Input((None, origin_units))
        z0 = Input((None, latent_units))
        mean, logvar = encoder.as_layer()(ni)
        def sample(data):
            z0, mean, logvar = data
            return mean + z0 * tf.exp(0.5 * logvar)
        z = Lambda(sample)([z0, mean, logvar])
        no = decoder.as_layer()(z)
        return tl.models.Model(inputs = [ni, z0], outputs = [mean, logvar, no])
    model_train = build_train_model()
    return encoder, decoder, model_train
