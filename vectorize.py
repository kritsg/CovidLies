import tensorflow as tf
import re
import string
import pandas as pd

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
def vectorize_col(col:pd.Series):
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')
    # Define the vocabulary size and number of words in a sequence.
    vocab_size = 4096
    sequence_length = 10
    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Set output_sequence_length length to pad all samples to same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(col.to_numpy())
    col = col.map(vectorize_layer)
    return 
