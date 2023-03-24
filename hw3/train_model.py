from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense
import tensorflow as tf

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()
    #model.add(...)
    model = tf.keras.Sequential()
    #the words having same embedding layer =6
    model.add(tf.keras.layers.Embedding(word_types, 32, input_length = 6)) 
    #flatten the output of embedding layer firts
    model.add(keras.layers.Flatten())
    #dense hidden layer of units = 100
    model.add(keras.layers.Dense(100, activation="relu"))
    #dense hidden layer of units=10
    model.add(keras.layers.Dense(10, activation="relu"))
    #output layer using softmax activation units=91
    model.add(keras.layers.Dense(91, activation='softmax'))
    # prepare the model for training
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")
    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load('data/input_train.npy')
    outputs = np.load('data/target_train.npy')
    print("Done loading data.")
   
    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)
    
    model.save('data/model.h5')
