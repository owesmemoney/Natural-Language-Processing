from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: # TODO: Write the body of this loop for part 4 
            input_rep = self.extractor.get_input_representation(words, pos, state).reshape((1, 6))
            pre = self.model.predict(input_rep)[0]
            index = list(np.argsort(pre)[::-1])
            for i in index:
                trs, label = self.output_labels[i]
                # transition = shift: len(buffer) >=1; stack is empty
                if trs == 'shift':
                    if len(state.buffer) >=1 or not state.stack:
                        state.shift()
                        break
                # transition = left_arc; root is not second item on stack and len(stack)>0
                elif trs == 'left_arc':
                    if state.stack[-1]==0 or not state.stack:
                        continue
                    else:
                        state.left_arc(label)
                        break
                # transition = right_arc; len(stack) not empty
                else:
                    if state.stack:
                        state.right_arc(label)
                        break
                    

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

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
    parser = Parser(extractor, 'data/model.h5')

    with open('data/dev.conll','r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
