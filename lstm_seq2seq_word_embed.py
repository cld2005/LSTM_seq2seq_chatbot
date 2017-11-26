from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'conv/codedak_conv.txt'
BASE_DIR = '../'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 2000
EMBEDDING_DIM = 100
NUM_PREDICTION = 50
TRIANABLE = True
START_SIGN = '*'
SPACE = ' '
STOP_SIGN = '.'
FILTER_STRING = '$!"#%&()+,-/:;<=>?@[\\]^_`{|}~'

print('loading Glove.6B 100 embedding.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

input_texts = []
target_texts = []

lines = open(data_path).read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "*" as the "start sequence" character
    # for the targets, and "." as "end sequence" character.

    input_text_words = input_text.split(SPACE)
    input_text_words = [word for word in input_text_words if
                        embeddings_index.has_key(word)]  # take only words in the word embedding
    input_text = SPACE.join(input_text_words[:])
    target_text_words = target_text.split(SPACE)
    target_text_words = [word for word in target_text_words if
                         embeddings_index.has_key(word)]  # take only words in the word embedding
    target_len = len(target_text_words)
    if target_len > MAX_SEQUENCE_LENGTH:
        target_text = SPACE.join(target_text_words[0:MAX_SEQUENCE_LENGTH])
    else:
        target_text = SPACE.join(target_text_words[:])

    target_text = START_SIGN + SPACE + target_text + SPACE + STOP_SIGN
    input_texts.append(input_text)
    target_texts.append(target_text)

input_tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=FILTER_STRING)
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(
    input_texts)  # nothing in the sequence ls larger than MAX_NB_WORDS-1
input_word_index = input_tokenizer.word_index
# print('input_word_index: ', input_word_index)
# print('input_sequences: ', input_sequences)

target_tokenizer = Tokenizer(num_words=MAX_NB_WORDS + 2, filters=FILTER_STRING)
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_word_index = target_tokenizer.word_index
# print('target_word_index: ', target_word_index)
# print('target_sequences: ', target_sequences)

num_encoder_tokens = min(len(input_word_index), MAX_NB_WORDS) + 1
num_decoder_tokens = min(len(target_word_index), MAX_NB_WORDS + 2) + 1
max_encoder_seq_length = max([len(txt) for txt in input_sequences])
max_decoder_seq_length = max([len(txt) for txt in target_sequences])

input_sequences = pad_sequences(input_sequences, padding='post',
                                maxlen=min(max_encoder_seq_length, MAX_SEQUENCE_LENGTH), truncating='post');
target_sequences = pad_sequences(target_sequences, padding='post',
                                 maxlen=min(max_decoder_seq_length, MAX_SEQUENCE_LENGTH + 2), truncating='post');

print('Number of samples:', len(input_texts))
print('Number of encoder input:', len(input_sequences))
print('Number of decoder input:', len(target_sequences))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

print('Preparing embedding matrix.')

# prepare input embedding matrix
print('prepare input embedding matrix')
# input_num_words = num_encoder_tokens
input_embedding_matrix = np.zeros((num_encoder_tokens, EMBEDDING_DIM))
hit_count = 0
for word, i in input_word_index.items():
    # print(word, i)
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        input_embedding_matrix[i] = embedding_vector
        hit_count = hit_count + 1

print("hit_count :", hit_count, "percentage: ", hit_count * 1.0 / (num_encoder_tokens))
# prepare decoder embedding matrix
print('prepare decoder embedding matrix')
# decoder_num_words = num_decoder_tokens
decoder_embedding_matrix = np.zeros((num_decoder_tokens, EMBEDDING_DIM))

for word, i in target_word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        decoder_embedding_matrix[i] = embedding_vector

# Define an input sequence and process it.
print('Define an input sequence and process it')
encoder_inputs = Input(shape=(None,))

x = Embedding(num_encoder_tokens,
              EMBEDDING_DIM,
              weights=[input_embedding_matrix],
              trainable=TRIANABLE,
              mask_zero=True)(encoder_inputs)

encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens,
                              EMBEDDING_DIM,
                              weights=[decoder_embedding_matrix],
                              trainable=TRIANABLE,
                              mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_lstm_output)
print('decoder_outputs: ', decoder_outputs)

encoder_input_data = np.asarray(input_sequences)
decoder_input_data = np.asarray(target_sequences)

decoder_target_data = np.zeros(
    (len(input_texts), min(max_decoder_seq_length, MAX_SEQUENCE_LENGTH + 2), num_decoder_tokens),
    dtype='float32')

for i, target_sequence in enumerate(target_sequences):
    for t, num in enumerate(target_sequence):
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            # print ('num',num)
            decoder_target_data[i, t - 1, num] = 1.

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.load_weights('s2s_south_park.h5')
print('start fitting')
print("encoder_input_data ", encoder_input_data.shape)
print("decoder_input_data ", decoder_input_data.shape)
print("decoder_target_data ", decoder_target_data.shape)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s_south_park_100_trainable.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

x, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

x = decoder_dense(x)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [x] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, word) for word, i in input_word_index.items())
reverse_input_char_index[0] = ''
reverse_target_char_index = dict(
    (i, word) for word, i in target_word_index.items())
reverse_target_char_index[0] = ''


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_list = list()
    target_list.append(target_word_index[START_SIGN])
    target_seq = np.asarray(target_list)
    target_seq = target_seq.reshape(1, target_seq.shape[0])
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    first_draw = True

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        if first_draw:
            sampled_token_index = np.random.choice(np.size(output_tokens[0, -1, :]), 1, p=output_tokens[0, -1, :])
            sampled_token_index = sampled_token_index[0];
            first_draw = False
        else:
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("sampled_token_index", sampled_token_index, "p= ", output_tokens[0, -1, sampled_token_index])
        sampled_word = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_word)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == STOP_SIGN or
                    len(decoded_sentence) > min(max_decoder_seq_length, MAX_SEQUENCE_LENGTH + 2)):
            stop_condition = True

        # Update the target sequence
        target_list.append(sampled_token_index)
        target_seq = np.asarray(target_list)
        target_seq = target_seq.reshape(1, target_seq.shape[0])
        print("target_seq shape", target_seq.shape)
        # Update states
        states_value = [h, c]

    return SPACE.join(decoded_sentence)


for seq_index in range(NUM_PREDICTION):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    print('---------------------------------------')
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
