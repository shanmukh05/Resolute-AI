import tensorflow as tf
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def BinaryLoss(y_true, y_pred, label_smoothing=0., reduce_dim=True, from_logits=True):
    y_pred = tf.Variable(y_pred, dtype=tf.float32)
    y_true = tf.Variable(y_true, dtype=tf.float32)

    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing       # apply label smoothing

    if from_logits:                  # if from_logits =True, convert them to (0,1) range from (-inf,+inf)
        y_pred_exp = tf.math.exp(-1 * y_pred)
        y_pred = 1 / (1 + y_pred_exp)

    if len(y_pred.shape) == 2:
        batch_size = y_pred.shape[1]
        batch_size = tf.cast(batch_size, tf.float32)
        N = y_pred.shape[0] * y_pred.shape[1]
        N = tf.cast(N, tf.float32)
    elif len(y_pred.shape) == 1:
        N = y_pred.shape[0]
        N = tf.cast(N, tf.float32)
    else:             # if shape of input tensor is wrong then exit the code
        loss = "Wrong Input"
        print("Error in input shape")
        sys.exit()

    y_pred_log = tf.math.log(y_pred)
    y_pred_minus_log = tf.math.log(1 - y_pred)

    if reduce_dim:              # Set the dimension of output loss according to reduce_dim attribute
        if len(y_pred.shape) == 2:
            loss_val = -1 / N * (y_true * y_pred_log + (1 - y_true) * y_pred_minus_log)      # Binary Cross Entropy loss function equation
            loss = tf.math.reduce_sum(loss_val, axis=[0, 1])

        else:
            loss_val = -1 / N * (y_true * y_pred_log + (1 - y_true) * y_pred_minus_log)
            loss = tf.math.reduce_sum(loss_val, axis=-1)

    else:
        if len(y_pred.shape) == 2:
            loss_val = -1 / batch_size * (y_true * y_pred_log + (1 - y_true) * y_pred_minus_log)
            loss = tf.math.reduce_sum(loss_val, axis=-1)

        else:
            loss = -1 * (y_true * y_pred_log + (1 - y_true) * y_pred_minus_log)

    return loss


print("------Test 1------")
y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]
y_true = tf.constant(y_true, dtype=tf.float32)
y_pred = tf.Variable(y_pred, dtype=tf.float32)
loss1 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=False)
loss2 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=True)
print("Binary Cross Entropy Loss Custom(without dimension reduction): ", loss1)
print("Binary Cross Entropy Loss Custom(with dimension reduction): ", loss2)

print("------Test 2------")
y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]
y_true = tf.constant(y_true, dtype=tf.float32)
y_pred = tf.Variable(y_pred, dtype=tf.float32)
loss1 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=False)
loss2 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=True)
print("Binary Cross Entropy Loss Custom:(without dimension reduction): ", loss1)
print("Binary Cross Entropy Loss Custom:(with dimension reduction): ", loss2)


print("------Test 3------")
y_true = [[[0, 1], [0, 0]]]
y_pred = [[[-18.6, 0.51], [2.94, -12.8]]]
y_true = tf.constant(y_true, dtype=tf.float32)
y_pred = tf.Variable(y_pred, dtype=tf.float32)
loss1 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=False)
loss2 = BinaryLoss(y_true, y_pred, from_logits=True, reduce_dim=True)
print("Binary Cross Entropy Loss Custom:(without dimension reduction): ", loss1)
print("Binary Cross Entropy Loss Custom:(with dimension reduction): ", loss2)
