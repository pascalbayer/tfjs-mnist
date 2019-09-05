import * as tf from '@tensorflow/tfjs';
import { IMAGE_H, IMAGE_W } from './data';

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
export function createConvModel(): tf.Sequential {
    const model = tf.sequential();

    model.add(
        tf.layers.conv2d({
            inputShape: [IMAGE_H, IMAGE_W, 1],
            kernelSize: 5,
            filters: 32,
            activation: 'relu'
        })
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    model.add(
        tf.layers.conv2d({
            inputShape: [IMAGE_H, IMAGE_W, 1],
            kernelSize: 5,
            filters: 64,
            activation: 'relu'
        })
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    model.add(tf.layers.flatten({}));

    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    return model;
}
