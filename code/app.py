from typing import List, Dict, Tuple

import gensim
import tensorflow as tf
from nltk import FreqDist
from tensorflow.python import keras as k

import config
import models
import parser
import plotter
import predict
import preprocesser
import utils

"""
    :author Silvio Severino
"""


def preprocessing_domain(
    train_y: List[str],
    dev_y: List[str],
    dom_path: str,
    to_write: str,
    vocab: Dict[str, int],
    num_vocab: int,
    min_count: int,
    reverse: bool = False,
    bn: bool = True,
) -> Tuple[List[str], List[str], Dict[str, int], FreqDist]:
    """
    This method is used to preprocess the labels.

    :param train_y: the labels to preprocess
    :param dev_y: the dev labels to preprocess
    :param dom_path: the path which from read the conversion map
    :param to_write: the path to write the vocabulary
    :param vocab: training vocabulary
    :param num_vocab: maximum number of vocab
    :param min_count: min number of word occurrences
    :param reverse: if True, it reads domain2bn. False otherwise
    :param bn: if True is done the Babelnet preprocess. False coarse-grained
    :return:
        train_y = train_y preprocessed
        dev_y = dev_y preprocessed
        out_vocab = the sense-inventory
        frequency = word frequency
    """
    bn2domain = utils.read_map(dom_path, reverse=False)
    if not bn:
        train_y, dev_y = utils.domains_converters([train_y, dev_y], bn2domain)

    bn2domain = utils.read_map(dom_path, reverse=reverse)
    senses = utils.retrieve_senses(train_y, bn2domain, is_bn=bn)

    out_vocab, frequency = preprocesser.compute_vocabulary(
        senses, min_count=min_count, num_vocab=num_vocab, vocab=vocab
    )

    train_y, dev_y = preprocesser.text2id_corpus(
        corpus=[train_y, dev_y], vocab=out_vocab
    )

    utils.write_vocabulary(to_write, out_vocab)

    return train_y, dev_y, out_vocab, frequency


def preprocessing(filepath: str) -> Tuple[List[str], List[str], List[str]]:
    """
    This method is used to read the preprocessed training sentences
    :param filepath: the path from read
    :return:
        sentences = train_x
        labels = train_y
        pos = pos sentences
    """

    sentences, labels, pos = utils.read_sentences_and_labels(filepath)

    # check whether the length are the same
    print("\t\tCheck Integrity", utils.check_integrity(sentences, labels))

    return sentences, labels, pos


def train(
    inp_tr: List,
    inp_dev: List,
    inp_sens: List,
    inp_sense_dev: List,
    inp_pos: List,
    dev_pos: List,
    lab_bn: List,
    lab_dev_bn: List,
    lab_wnd: List,
    lab_dev_wnd: List,
    lab_lex: List,
    lab_dev_lex: List,
    lab_pos: List,
    dev_lab_pos: List,
    pre_trained: gensim.models.word2vec.Word2Vec,
    vocab: Dict[str, int],
    out_vocab_bn: Dict[str, int],
    out_vocab_wnd: Dict[str, int],
    out_vocab_lex: Dict[str, int],
    out_vocab_pos: Dict[str, int],
    restore: bool = False,
    initial_epoch: int = 0,
) -> k.Model:
    """
    Training method
    :param inp_tr: training input
    :param inp_dev: development input
    :param inp_sens: sense_embeddings phrase input
    :param inp_sense_dev: development sense_embeddings phrase input
    :param inp_pos: pos input
    :param dev_pos: development input
    :param lab_bn: babelnet labels
    :param lab_dev_bn: development babelnet labels
    :param lab_wnd: wordnet domains labels
    :param lab_dev_wnd: development wordnet domains labels
    :param lab_lex: lexicographer labels
    :param lab_dev_lex: development lexicographer labels
    :param lab_pos: pos labels
    :param dev_lab_pos: development pos labels
    :param pre_trained: pretrained embeddings
    :param vocab: training vocab
    :param out_vocab_bn: babelnet sense inventory
    :param out_vocab_wnd: wordnet domains sense inventory
    :param out_vocab_lex: lexicographer sense inventory
    :param out_vocab_pos: pos sense inventory
    :param restore: if True, it restores the weights model.
                        False, it trains a new model
    :param initial_epoch: initial epoch from restore
    :return: keras Model
    """
    batch_size = 64
    epochs = 40
    pad = 25

    check_point = tf.keras.callbacks.ModelCheckpoint(
        str(config.MODEL_CHECK_POINT),
        monitor="val_dense_bn_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_dense_bn_loss", patience=4, mode="min", verbose=1
    )

    if config.SENSE_EMB:
        generator = models.batch_generator_sens_emb(
            inp_tr,
            inp_sens,
            inp_pos,
            lab_bn,
            lab_wnd,
            lab_lex,
            batch_size,
            pad,
            is_elmo=config.IS_ELMO,
        )

        val_generator = models.batch_generator_sens_emb(
            inp_dev,
            inp_sense_dev,
            dev_pos,
            lab_dev_bn,
            lab_dev_wnd,
            lab_dev_lex,
            batch_size,
            pad,
            is_elmo=config.IS_ELMO,
        )
    else:
        generator = models.batch_generator(
            inp_tr,
            lab_bn,
            lab_wnd,
            lab_lex,
            lab_pos,
            batch_size,
            pad,
            is_elmo=config.IS_ELMO,
        )

        val_generator = models.batch_generator(
            inp_dev,
            lab_dev_bn,
            lab_dev_wnd,
            lab_dev_lex,
            dev_lab_pos,
            batch_size,
            pad,
            is_elmo=config.IS_ELMO,
        )

    model = models.build_model(
        vocab_size=len(vocab),
        out_size_bn=len(out_vocab_bn),
        out_size_wnd=len(out_vocab_wnd),
        out_size_lex=len(out_vocab_lex),
        out_size_pos=len(out_vocab_pos),
        word2vec=pre_trained,
        is_elmo=config.IS_ELMO,
        attention=config.ATTENTION,
        is_sense_emb=config.SENSE_EMB,
    )

    if restore:
        model.load_weights(str(config.MODEL_CHECK_POINT))

    history = model.fit_generator(
        generator,
        steps_per_epoch=len(inp_tr) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(inp_dev) // batch_size,
        callbacks=[check_point, early_stopping],
        initial_epoch=initial_epoch,
    )

    plotter.plot_train(history)

    model.save_weights(str(config.MODEL_WEIGHTS_TR))
    model.save(config.MODEL_TRY)
    return model


def processing(
    train_x: List[str],
    train_y: List[str],
    dev_x: List[str],
    dev_y: List[str],
    pos_tr: List[str],
    pos_dev: List[str],
) -> None:
    """
    This method is used to process all the corpus, preparing it for the model
    :param train_x: the sentences to preprocess
    :param train_y: the labels to preprocess
    :param dev_x: the dev sentences to preprocess
    :param dev_y: the dev labels to preprocess
    :param pos_tr: the pos sentences to preprocess
    :param pos_dev: the dev pos sentences to preprocess
    :return: None
    """
    num_vocab_x = 11000
    num_vocab_y = None
    min_count = 0

    tr_sens = utils.convert_sens_emb(train_x, train_y)
    dev_sens = utils.convert_sens_emb(dev_x, dev_y)

    vocab, emb_vocab, emb, vocab_pos, frequency_tr, frequency_pos = preprocesser.compute_train_vocab(
        train_x, tr_sens, pos_tr, min_count=min_count, num_vocab=num_vocab_x
    )

    tmp_pos, tmp_dev_pos = pos_tr, pos_dev
    if not config.IS_ELMO:
        train_x, dev_x = preprocesser.text2id_corpus(
            corpus=[train_x, dev_x], vocab=vocab
        )
        pos_tr, pos_dev = preprocesser.text2id_corpus(
            corpus=[pos_tr, pos_dev], vocab=vocab_pos
        )

    tr_sens, dev_sens = preprocesser.text2id_corpus(
        corpus=[tr_sens, dev_sens], vocab=emb_vocab
    )

    print("\nProcessing domains...")

    print("\tBabelnet")
    lab_bn, dev_lab_bn, sens_inv_bn, frequency_bn = preprocessing_domain(
        train_y,
        dev_y,
        dom_path=config.BABELNET2WNDOMAINS_TR,
        vocab=vocab,
        to_write=config.OUT_VOCAB_BN_TR,
        min_count=min_count,
        num_vocab=num_vocab_y,
        reverse=False,
    )

    print("\tWordnet domain")
    lab_wnd, dev_lab_wnd, sens_inv_wnd, frequency_wnd = preprocessing_domain(
        train_y,
        dev_y,
        dom_path=config.BABELNET2WNDOMAINS_TR,
        vocab=None,
        to_write=config.OUT_VOCAB_WND_TR,
        min_count=0,
        num_vocab=num_vocab_y,
        reverse=True,
        bn=False,
    )

    print("\tLexicographer")
    lab_lex, dev_lab_lex, sens_inv_lex, frequency_lex = preprocessing_domain(
        train_y,
        dev_y,
        dom_path=config.BABELNET2LEXANAMES_TR,
        vocab=None,
        to_write=config.OUT_VOCAB_LEX_TR,
        min_count=0,
        num_vocab=num_vocab_y,
        reverse=True,
        bn=False,
    )

    print("\tPart of speech")
    lab_pos, dev_lab_pos = preprocesser.text2id_corpus(
        corpus=[tmp_pos, tmp_dev_pos], vocab=vocab_pos
    )

    plotter.plot_corpus_train(
        train_x,
        dev_x,
        tr_sens,
        pos_tr,
        vocab,
        lab_bn,
        dev_lab_bn,
        sens_inv_bn,
        lab_wnd,
        dev_lab_wnd,
        sens_inv_wnd,
        lab_lex,
        dev_lab_lex,
        sens_inv_lex,
        lab_pos,
        dev_lab_pos,
        vocab_pos,
    )

    plotter.plot_frequencies(
        [
            (frequency_tr, "Train_x"),
            (frequency_bn, "Babelnet labels"),
            (frequency_wnd, "Wordnet domains labels"),
            (frequency_lex, "Lexicographer labels"),
            (frequency_pos, "POS"),
        ]
    )

    train(
        train_x,
        dev_x,
        tr_sens,
        dev_sens,
        pos_tr,
        pos_dev,
        lab_bn,
        dev_lab_bn,
        lab_wnd,
        dev_lab_wnd,
        lab_lex,
        dev_lab_lex,
        lab_pos,
        dev_lab_pos,
        emb,
        vocab,
        sens_inv_bn,
        sens_inv_wnd,
        sens_inv_lex,
        vocab_pos,
    )


def main():

    print("Parsing all...")
    parser.parse_all()

    print("\nPreprocessing...")
    print("\tPreprocessing training set")
    train_x, train_y, pos_tr = preprocessing(config.SEMCOR_SENTENCES)
    print("\tPreprocessing dev set")
    dev_x, dev_y, pos_dev = preprocessing(config.SEMEVAL2007_SENTENCES)

    processing(train_x, train_y, dev_x, dev_y, pos_tr, pos_dev)

    predict.predicter()


if __name__ == "__main__":
    main()
