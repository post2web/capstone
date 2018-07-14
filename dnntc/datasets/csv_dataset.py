import os
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import multiprocessing


class Dataset(object):
    
    def __init__(self,
        filenames,
        vocab_file,
        max_text_length=400,
        batch_size=128,
        skip_header_lines=0,
        text_feature_name='review',
        target_name='class',
        weight_column_name='weight',
        pad_word='#@PAD@#',
        csv_header=['class', 'polarity', 'source', 'fold', 'file', 'review'],
        target_labels=['False', 'True'],
        num_epochs=None,
        multi_threading=True,
        prefetch=1,
        words_to_ids=True,
        shuffle=True
        ):
        
        #self.filenames = tf.gfile.Glob(filenames)
        self.filenames = filenames
        #'data/op_spam_v1.4/vocab.csv'
        with open(vocab_file) as f:
            self.n_words = sum(1 for line in f) + 2
            
        self.vocab_file = vocab_file
        self.csv_header = csv_header
        self.max_text_length=max_text_length
        self.text_feature_name=text_feature_name
        self.target_name=target_name
        self.weight_column_name=weight_column_name
        self.pad_word=pad_word
        self.target_labels=target_labels
        
        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        buffer_size = 2 * batch_size + 1

        print("")
        print("* data input_fn:")
        print("=" * 20)
        print("Input file(s): {}".format(self.filenames))
        print("Batch size: {}".format(batch_size))
        print("Epoch Count: {}".format(num_epochs))
        print("Thread Count: {}".format(num_threads))
        print("Shuffle: {}".format(shuffle))
        print("=" * 20)
        print("")


        dataset = tf.data.TextLineDataset(filenames=tf.matching_files(self.filenames))

        dataset = dataset.skip(skip_header_lines)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.map(
            lambda tsv_row: self.parse_csv_row(tsv_row), 
            num_parallel_calls=num_threads)

        if batch_size:
            dataset = dataset.batch(batch_size)
        
        if num_epochs:
            dataset = dataset.repeat(num_epochs)

        if prefetch:
            dataset = dataset.prefetch(prefetch)

        iterator = dataset.make_one_shot_iterator()

        features, target = iterator.get_next()
        target = self.labels_to_ids(target)
        
        if words_to_ids:
            features[text_feature_name] = self.words_to_ids(features[text_feature_name])
        
        self.features = features
        self.target = target
        
    def input_fn(self):
        return self.features, self.target
    
    def parse_csv_row(self, tsv_row):
        record_defaults = [['NA']] * len(self.csv_header)
        columns = tf.decode_csv(tsv_row, record_defaults=record_defaults)
        features = dict(zip(self.csv_header, columns))
        target = features.pop(self.target_name)
        features[self.weight_column_name] =  tf.cond(tf.equal(target, 'True'), lambda: 1.0, lambda: 1.0 )
        return features, target

    def labels_to_ids(self, label_string_tensor):
        table = tf.contrib.lookup.index_table_from_tensor(tf.constant(self.target_labels))
        return table.lookup(label_string_tensor)
    
    def words_to_ids(self, reviews):
        # Load vocabolary lookup table to map word => word_id
        vocab_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self.vocab_file,
            num_oov_buckets=1,
            default_value=-1
        )

        # Split text to words -> this will produce sparse tensor with variable-lengthes (word count) entries
        words = tf.string_split(reviews)
        # Convert sparse tensor to dense tensor by padding each entry to match the longest in the batch
        dense_words = tf.sparse_tensor_to_dense(words, default_value=self.pad_word)
        # Convert word to word_ids via the vocab lookup table
        word_ids = vocab_table.lookup(dense_words)
        # Create a word_ids padding
        padding = tf.constant([[0, 0],[0, self.max_text_length]])
        # Pad all the word_ids entries to the maximum document length
        word_ids_padded = tf.pad(word_ids, padding)
        word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, self.max_text_length])
        # Return the final word_id_vector
        return word_id_vector


