import codecs
import os
import pkg_resources
import pickle
import warnings

import numpy as np
import sklearn.metrics
import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from neuroner.evaluate import remap_labels
from neuroner import utils_tf
from neuroner import utils_nlp

def train_step(sess, dataset, sequence_number, model, parameters):
    """
    Train.
    """
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]

    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.UNK_TOKEN_INDEX

    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: 1-parameters['dropout_rate']}

    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy,
                    model.transition_parameters],feed_dict)

    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained,
                    epoch_number, parameters):
    """
    Predict.
    """
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))

    all_predictions = []

    for i in range(len(dataset.token_indices[dataset_type])):
        feed_dict = {
          model.input_token_indices: dataset.token_indices[dataset_type][i],
          model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
          model.input_token_lengths: dataset.token_lengths[dataset_type][i],
          model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
          model.dropout_keep_prob: 1.
        }

        unary_scores, predictions = sess.run([model.unary_scores,
            model.predictions], feed_dict)

        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores,
                transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert(len(predictions) == len(dataset.tokens[dataset_type][i]))

        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        all_predictions.extend(prediction_labels)

    return all_predictions


def predict_labels(sess, model, transition_params_trained, parameters, dataset,
    epoch_number, stats_graph_folder, dataset_filepaths):
    """
    Predict labels using trained model
    """
    y_pred = {}
    y_true = {}
    output_filepaths = {}

    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue

        prediction_output = prediction_step(sess, dataset, dataset_type, model,
            transition_params_trained, stats_graph_folder, epoch_number,
            parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output

    return y_pred, y_true, output_filepaths
