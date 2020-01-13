import tensorlayer as tl
import tensorflow as tf
import numpy as np
import config
from model import build
from data import enum_train, cnt_train, enum_val, cnt_val, gray2rgb

model_train, model_encoder, model_decoder = build(784, config.hidden_units, config.latent_units)
optimizer = tf.optimizers.Adam(learning_rate = config.learning_rate)

tl.files.exists_or_mkdir(config.save_snapshot_to)
tl.files.exists_or_mkdir(config.save_visualization_to)
writer = tf.summary.create_file_writer(config.save_logs_to)

def l2_loss(x, y):
    return tf.reduce_sum(tf.losses.binary_crossentropy(x, y))

def kl_loss(mean, logstdev):
    return -0.5 * tf.reduce_sum(1 + logstdev - tf.exp(logstdev) - mean ** 2)

for epoch in range(config.cnt_epoch):
    print('Epoch %d/%d' % (epoch, config.cnt_epoch))
    tf.summary.experimental.set_step(epoch)

    model_train.train()
    loss_kl_sum, loss_l2_sum, loss_sum = 0, 0, 0
    for inputs in enum_train():
        z0 = np.random.normal(0.0, 1.0, (inputs.shape[0], config.latent_units)).astype(np.float32)
        with tf.GradientTape() as tape:
            mean, logvar, outputs = model_train([inputs, z0])
            loss_kl = kl_loss(mean, logvar)
            loss_l2 = l2_loss(inputs, outputs)
            loss = loss_kl + loss_l2
        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        loss_kl_sum += loss_kl * inputs.shape[0] / cnt_train
        loss_l2_sum += loss_l2 * inputs.shape[0] / cnt_train
        loss_sum += loss * inputs.shape[0] / cnt_train

    with writer.as_default():
        tf.summary.scalar('train_loss_kl', loss_kl_sum)
        tf.summary.scalar('train_loss_l2', loss_l2_sum)
        tf.summary.scalar('train_loss', loss_sum)
        writer.flush()

    model_train.eval()
    loss_kl_sum, loss_l2_sum, loss_sum = 0, 0, 0
    for inputs in enum_train():
        z0 = np.random.normal(0.0, 1.0, (inputs.shape[0], config.latent_units)).astype(np.float32)
        mean, logvar, outputs = model_train([inputs, z0])
        loss_kl = kl_loss(mean, logvar)
        loss_l2 = l2_loss(inputs, outputs)
        loss = loss_kl + loss_l2

        loss_kl_sum += loss_kl * inputs.shape[0] / cnt_train
        loss_l2_sum += loss_l2 * inputs.shape[0] / cnt_train
        loss_sum += loss * inputs.shape[0] / cnt_train

    with writer.as_default():
        tf.summary.scalar('val_loss_kl', loss_kl_sum)
        tf.summary.scalar('val_loss_l2', loss_l2_sum)
        tf.summary.scalar('val_loss', loss_sum)
        writer.flush()

    if (epoch + 1) % config.log_every == 0:
        model_train.save_weights(os.path.join(config.save_snapshot_to, 'model_train_' + str(epoch + 1) + '.h5'))
        model_decoder.eval()
        z = np.random.normal(0.0, 1.0, (config.vis_count, config.latent_units)).astype(np.float32)
        outputs = model_decoder(z).numpy().reshape((-1, 28, 28))
        outputs = gray2rgb(outputs)
        im_path = os.path.join(config.save_visualization_to, 'test_' + str(epoch) + '.png')
        tl.visualize.save_images(outputs, config.vis_layout, im_path)
        with writer.as_default():
            tf.summary.image('visualize', np.expand_dims(tl.visualize.read_image(im_path), axis = 0))
            writer.flush()
    
