


def custom_loss(y_true, y_pred):
    y_true = tf.argmax(y_true, 3)

    #ll = categorical_crossentropy(logits, l)

    naive_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    # boot_loss, _ = tf.nn.top_k(tf.reshape(naive_loss, [tf.shape(im)[0], tf.shape(im)[1] * tf.shape(im)[2]]), k=K,
    #                               sorted=False)
    loss = tf.reduce_mean(tf.reduce_sum(naive_loss, axis=1))

    return loss


def weighted_crossentropy(y_true, y_pred):
    """
    variant of cross entropy loss function with weighting based on class labels
    """

    w = np.asarray([ 1.1, 40.0])
    w_tf = tf.constant(w,dtype=tf.float32)


    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, w_tf)


def tool_accuracy(y_true, y_pred):
    """
    Metric to evaluate foreground segmentation separately as the usual accuracy function considers both
    foreground and background. So to say the precision.

    Here, we inspect how many of the pixels that should be labeled as tool are so.
    Outliers are not considered.

    :param y_true:
    :param y_pred:
    :return:
    """

    y_pred_logical = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    mask = tf.equal(y_true,1)

    m = tf.boolean_mask(y_pred_logical, mask)
    return tf.reduce_mean(tf.reduce_mean(m))
