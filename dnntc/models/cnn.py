import tensorflow as tf

def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params.feature_columns)

    # 1d convolution
    words_conv = tf.layers.conv1d(net,
        filters=params.filters,
        kernel_size=params.window_size, 
        strides=params.window_size//2,
        padding='SAME',
        activation=tf.nn.relu)

    words_conv_shape = words_conv.get_shape()
    dim = words_conv_shape[1] * words_conv_shape[2]
    input_layer = tf.reshape(words_conv,[-1, dim])
    
    if params['hidden_units'] is not None:
        # optional fully-connected layer-stack
        hidden_layers = tf.contrib.layers.stack(
            inputs=input_layer,
            layer=tf.contrib.layers.fully_connected,
            stack_args=params['hidden_units'],
            activation_fn=tf.nn.relu
        )
    else:
        hidden_layers = input_layer

    # logits without activation
    logits = tf.layers.dense(
        inputs=hidden_layers, 
        units=len(TARGET_LABELS), 
        activation=None)
    
    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    # weights
    weights = features[WEIGHT_COLUNM_NAME]

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels, weights=weights)
    
    tf.summary.scalar('loss', loss)
    
    # TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step=tf.train.get_global_step()
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )
        
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities, weights=weights)
        }
        
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

