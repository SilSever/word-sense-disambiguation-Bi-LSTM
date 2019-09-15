from typing import List, Tuple

import gensim
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python import keras as k

import preprocesser

"""
    :author Silvio Severino
"""


def build_model(
    input_length: int = None,
    vocab_size: int = None,
    embedding_size: int = 200,
    hidden_size: int = 256,
    dropout: int = 0.8,
    recurrent_dropout: int = 0.6,
    out_size_bn: int = None,
    out_size_wnd: int = None,
    out_size_lex: int = None,
    out_size_pos: int = None,
    word2vec: gensim.models.word2vec.Word2Vec = None,
    is_sense_emb=False,
    is_elmo=False,
    attention=False,
) -> k.Model:
    """
    Build the model
    :param input_length: input length
    :param vocab_size: vocab size
    :param embedding_size: embedding size
    :param hidden_size: hidden size
    :param dropout: input droput
    :param recurrent_dropout: reccurrent dropout
    :param out_size_bn: Babelnet sense inventory size
    :param out_size_wnd: Wordnet domain sense inventory size
    :param out_size_lex: Lexicographer sense inventory size
    :param out_size_pos: POS sense inventory size
    :param word2vec: pre trained embeddings
    :param is_sense_emb: if True, it builds the sense embeddings model
                        if False, it build the normal model
    :param is_elmo: if True, it converts the input in ELMo format
                    if False, it uses keras embeddings
    :param attention: if True, it uses attention mechanism
    :return: a keras Model
    """

    input_layer = k.layers.Input(
        shape=(input_length,), name="main_input", dtype=("string" if is_elmo else None)
    )

    if is_elmo:
        embeddings = k.layers.Lambda(ELMoEmbedding, name="elmo")(input_layer)
    else:
        embeddings = k.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length,
            mask_zero=True,
            name="in_embeddings",
        )(input_layer)

    if is_sense_emb:
        embeddings, senses, pos = _get_sense_embeddings_layers(embeddings, word2vec)

    bi_lstm1 = k.layers.Bidirectional(
        k.layers.LSTM(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=False,
            name="bidirectional_1",
        )
    )(embeddings)
    lstm_out, for_hs, _, back_hs, _ = k.layers.Bidirectional(
        k.layers.LSTM(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
            name="bidirectional_2",
        )
    )(bi_lstm1)

    if attention:
        lstm_out = _get_attention_layer(lstm_out, is_elmo)

    lstm_out = k.layers.Dropout(0.5)(lstm_out)

    output_bn = k.layers.Dense(out_size_bn, activation="softmax", name="dense_bn")(
        lstm_out
    )
    output_wnd = k.layers.Dense(out_size_wnd, activation="softmax", name="dense_wnd")(
        lstm_out
    )
    output_lex = k.layers.Dense(out_size_lex, activation="softmax", name="dense_lex")(
        lstm_out
    )
    if not is_sense_emb:
        output_pos = k.layers.Dense(
            out_size_pos, activation="softmax", name="dense_pos"
        )(lstm_out)

    inputs = [input_layer, senses, pos] if is_sense_emb else input_layer
    outputs = (
        [output_bn, output_wnd, output_lex]
        if is_sense_emb
        else [output_bn, output_wnd, output_lex, output_pos]
    )

    model = k.Model(inputs=inputs, outputs=outputs)

    # start the session for ELMo
    sess = k.backend.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=k.optimizers.Adam(lr=0.0002),
        metrics=["acc"],
    )

    model.summary()
    return model


def batch_generator(
    feature: List,
    label_bn: List[List[int]],
    label_wnd: List[List[int]],
    label_lex: List[List[int]],
    label_pos: List[List[int]],
    batch_size: int,
    max_padding: int,
    is_elmo: bool = False,
) -> Tuple[List, List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """
    This method is used to compute the batch for the model
    :param feature: training sentences
    :param label_bn: Babelnet labels
    :param label_wnd: Wordnet domains labels
    :param label_lex: Lexicographer labels
    :param label_pos: Pos labels
    :param batch_size: the size of batch
    :param max_padding: maximum padding
    :param is_elmo: if True, it converts the feature in ELMo format
                    if False, it converts the feature with text2id
    :return: the batches
    """
    while True:
        for start in range(0, len(feature), batch_size):
            end = start + batch_size

            batch_x = feature[start:end]
            pad = min(max_padding, len(max(label_bn[start:end], key=len)))

            if is_elmo:
                batch_x_tokenized = [sent.split() for sent in batch_x]
                batch_x = preprocesser.padding_string(batch_x_tokenized, pad=pad)
            else:
                batch_x = preprocesser.padding(batch_x, pad=pad)

            batch_bn, batch_wnd, batch_lex, batch_pos = _batch_label(
                labels=[label_bn, label_wnd, label_lex, label_pos],
                start=start,
                end=end,
                pad=pad,
            )

            yield batch_x, [batch_bn, batch_wnd, batch_lex, batch_pos]


def batch_generator_sens_emb(
    feature: List,
    sens: List,
    pos: List,
    label_bn: List[List[int]],
    label_wnd: List[List[int]],
    label_lex: List[List[int]],
    batch_size: int,
    max_padding: int,
    is_elmo=False,
) -> Tuple[List, List[List[int]], List[List[int]], List[List[int]]]:
    """
    This method is used to compute the batch for the sense embedding model
    :param feature: training sentences
    :param sens: sense embeddings sentences
    :param pos: pos sentences
    :param label_bn: Babelnet labels
    :param label_wnd: Wordnet domains labels
    :param label_lex: Lexicographer labels
    :param batch_size: the size of batch
    :param max_padding: maximum padding
    :param is_elmo: if True, it converts the feature in ELMo format
                    if False, it converts the feature with text2id
    :return: the batches
    """
    while True:
        for start in range(0, len(feature), batch_size):
            end = start + batch_size

            batch_x = feature[start:end]
            pad = min(max_padding, len(max(label_bn[start:end], key=len)))

            if is_elmo:
                batch_x_tokenized = [sent.split() for sent in batch_x]
                batch_x = preprocesser.padding_string(batch_x_tokenized, pad=pad)
            else:
                batch_x = preprocesser.padding(batch_x, pad=pad)

            batch_sens = sens[start:end]
            batch_sens = preprocesser.padding(batch_sens, pad=pad)

            batch_pos = pos[start:end]
            batch_pos_tokenized = [p.split() for p in batch_pos]
            batch_pos = preprocesser.padding_string(batch_pos_tokenized, pad=pad)

            batch_bn, batch_wnd, batch_lex = _batch_label(
                labels=[label_bn, label_wnd, label_lex], start=start, end=end, pad=pad
            )

            yield [batch_x, batch_sens, batch_pos], [batch_bn, batch_wnd, batch_lex]


def _batch_label(
    labels: List[List[List[int]]], start: int, end: int, pad: int
) -> List[List[int]]:
    """
    This method is used to compute the batch for the labels
    :param labels: labels
    :param start: start index
    :param end: end index
    :param pad: maximum padding
    :return: a batch
    """
    for label in labels:

        batch_y = label[start:end]
        batch_y = preprocesser.padding(batch_y, pad=pad)
        batch_y = np.expand_dims(batch_y, -1)
        yield batch_y


def _get_sense_embeddings_layers(in_embeddings, word2vec):
    """
    This method returns the sense embeddings layer with sense embedding and pos
    :param in_embeddings: the normal input embeddings
    :param word2vec: pretrained sense embeddings
    :return:
            - the concatenated layer,
            - sense embeddings input layer
            - pos input layer
    """
    senses = k.layers.Input(shape=(None,), name="senses")
    sens_embeddings = _get_keras_embedding(word2vec, name="sens_emb")(senses)

    pos = k.layers.Input(shape=(None,), name="pos", dtype="string")
    pos_embeddings = k.layers.Lambda(ELMoEmbedding, name="elmo_pos")(pos)

    return (
        k.layers.Concatenate(name="embeddings")(
            [in_embeddings, sens_embeddings, pos_embeddings]
        ),
        senses,
        pos,
    )


class WeightedSum(k.layers.Layer):
    """
    A keras layer used to compute the Weighted sum between
    two layers usin a scalar
    """

    def __init__(self, a, **kwargs):
        self.a = a  # "weight" of the weighted sum
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def _get_attention_layer(bi_lstm, is_elmo: bool):
    """
    Computes the attention layer from the BiLSTM output
    :param bi_lstm: bilstm matrix
    :param is_elmo: If False, it computes the layer without the weighted sum
    :return: computed layer
    """
    attention = k.layers.TimeDistributed(
        k.layers.Dense(1, activation="tanh"), name="td_attention"
    )(bi_lstm)
    attention = k.layers.Activation("softmax", name="act_attention")(attention)
    if is_elmo:
        attention = WeightedSum(0.1)([bi_lstm, attention])
    return k.layers.Multiply(name="mul_attention")([bi_lstm, attention])


def _get_attention_layer_hidden(bi_lstm):
    """
    Computes the attention layer from the BiLSTM hidden state
    :param bi_lstm: bilstm matrix
    :return: computed layer
    """
    hidden = k.layers.Concatenate()([bi_lstm[1], bi_lstm[3]])
    hidden = k.layers.Activation("tanh")(hidden)
    hidden = k.layers.RepeatVector(k.backend.shape(bi_lstm[0])[1])(hidden)
    unit = k.layers.TimeDistributed(k.layers.Dense(1))(hidden)
    attention = k.layers.Activation("softmax")(unit)
    concat = WeightedSum(0.1)([hidden, attention])

    return k.layers.Multiply(name="mul_attention")([bi_lstm[0], concat])


def _get_keras_embedding(
    word2vec: gensim.models.word2vec.Word2Vec, name: str, train_embeddings: bool = False
):
    """
    Return a Tensorflow Keras 'Embedding' layer with weights set as the
    Word2Vec model's learned word embeddings.
    :param word2vec: a gensim Word2Vec model
    :param train_embeddings:
                      if False, the weights are frozen and stopped from being updated.
                      If True, the weights can/will be further trained/updated.
    :return: an Embedding layer.
    """
    weights = word2vec.wv.vectors

    pad = np.random.rand(1, weights.shape[1])
    unk = np.mean(weights, axis=0, keepdims=True)
    weights = np.concatenate((pad, unk, weights))

    layer = k.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
        mask_zero=True,
        name=name,
    )
    return layer


def ELMoEmbedding(x):
    """
    Elmo Embeddings
    :param x: tensor to convert
    :return: converted tensor
    """
    embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    x = tf.cast(x, tf.string)
    return embed(tf.reshape(x, [-1]), signature="default", as_dict=True)["elmo"]
