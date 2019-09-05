/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
import { MnistData } from './data';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui';
import { createConvModel } from './model';

tf.setBackend('webgl');

/**
 * This callback type is used by the `train` function for insertion into
 * the model.fit callback loop.
 *
 * @callback onIterationCallback
 * @param {string} eventType Selector for which type of event to fire on.
 * @param {number} batchOrEpochNumber The current epoch / batch number
 * @param {tf.Logs} logs Logs to append to
 */

/**
 * Compile and train the given model.
 *
 * @param {tf.Model} model The model to train.
 * @param {onIterationCallback} onIteration A callback to execute every 10
 *     batches & epoch end.
 */
async function train(model, onIteration) {
    ui.logStatus('Training model...');

    // Now that we've defined our model, we will define our optimizer. The
    // optimizer will be used to optimize our model's weight values during
    // training so that we can decrease our training loss and increase our
    // classification accuracy.

    // We are using rmsprop as our optimizer.
    // An optimizer is an iterative method for minimizing an loss function.
    // It tries to find the minimum of our loss function with respect to the
    // model's weight parameters.
    const optimizer = 'rmsprop';

    // We compile our model by specifying an optimizer, a loss function, and a
    // list of metrics that we will use for model evaluation. Here we're using a
    // categorical crossentropy loss, the standard choice for a multi-class
    // classification problem like MNIST digits.
    // The categorical crossentropy loss is differentiable and hence makes
    // model training possible. But it is not amenable to easy interpretation
    // by a human. This is why we include a "metric", namely accuracy, which is
    // simply a measure of how many of the examples are classified correctly.
    // This metric is not differentiable and hence cannot be used as the loss
    // function of the model.
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Batch size is another important hyperparameter. It defines the number of
    // examples we group together, or batch, between updates to the model's
    // weights during training. A value that is too low will update weights using
    // too few examples and will not generalize well. Larger batch sizes require
    // more memory resources and aren't guaranteed to perform better.
    const batchSize = 320;

    // Leave out the last 15% of the training data for validation, to monitor
    // overfitting during training.
    const validationSplit = 0.15;

    // Get number of training epochs from the UI.
    const trainEpochs = ui.getTrainEpochs();

    // We'll keep a buffer of loss and accuracy values over time.
    let trainBatchCount = 0;

    const trainData = data.getTrainData();
    const testData = data.getTestData();

    const totalNumBatches =
        Math.ceil((trainData.xs.shape[0] * (1 - validationSplit)) / batchSize) *
        trainEpochs;

    // During the long-running fit() call for model training, we include
    // callbacks, so that we can plot the loss and accuracy values in the page
    // as the training progresses.
    let valAcc;
    await model.fit(trainData.xs, trainData.labels, {
        batchSize,
        validationSplit,
        epochs: trainEpochs,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                trainBatchCount++;
                ui.logStatus(
                    `Training... (` +
                        `${((trainBatchCount / totalNumBatches) * 100).toFixed(
                            1
                        )}%` +
                        ` complete). To stop training, refresh or close page.`
                );
                ui.plotLoss(trainBatchCount, logs.loss, 'train');
                ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
                if (onIteration && batch % 10 === 0) {
                    onIteration('onBatchEnd', batch, logs);
                }
                await tf.nextFrame();
            },
            onEpochEnd: async (epoch, logs) => {
                valAcc = logs.val_acc;
                ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
                ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
                if (onIteration) {
                    onIteration('onEpochEnd', epoch, logs);
                }
                await tf.nextFrame();
            }
        }
    });

    const testResult = model.evaluate(testData.xs, testData.labels);
    const testAccPercent = testResult[1].dataSync()[0] * 100;
    const finalValAccPercent = valAcc * 100;
    ui.logStatus(
        `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
            `Final test accuracy: ${testAccPercent.toFixed(1)}%`
    );
}

/**
 * Show predictions on a number of test examples.
 *
 * @param {tf.Model} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
    const testExamples = 100;
    const examples = data.getTestData(testExamples);

    // Code wrapped in a tf.tidy() function callback will have their tensors freed
    // from GPU memory after execution without having to call dispose().
    // The tf.tidy callback runs synchronously.
    tf.tidy(() => {
        const output = model.predict(examples.xs);

        // tf.argMax() returns the indices of the maximum values in the tensor along
        // a specific axis. Categorical classification tasks like this one often
        // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
        // one element for each output class. All values in the vector are 0
        // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
        // output from model.predict() will be a probability distribution, so we use
        // argMax to get the index of the vector element that has the highest
        // probability. This is our prediction.
        // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
        // dataSync() synchronously downloads the tf.tensor values from the GPU so
        // that we can use them in our normal CPU JavaScript code
        // (for a non-blocking version of this function, use data()).
        const axis = 1;
        const labels = Array.from(examples.labels.argMax(axis).dataSync());
        const predictions = Array.from(output.argMax(axis).dataSync());

        ui.showTestResults(examples, predictions, labels);
    });
}

function createModel() {
    return createConvModel();
}

let data;
async function load() {
    data = new MnistData();
    await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
ui.setTrainButtonCallback(async () => {
    ui.logStatus('Loading MNIST data...');
    await load();

    ui.logStatus('Creating model...');
    const model = createModel();
    model.summary();

    ui.logStatus('Starting model training...');
    await train(model, () => showPredictions(model));

    await model.save('downloads://model');
});
