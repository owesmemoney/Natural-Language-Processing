#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

import gensim
import transformers 

from typing import List

import string
from collections import defaultdict


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1; takes a lemma and part of speech; return the possible substitutes based on WordNet
    all_sets = wn.synsets(lemma, pos=pos) # function constrain the part of speech
    result = []
    for i in all_sets:
        for j in i.lemma_names():#find lematized words
            if j not in result and j != lemma:# output not contain lemma
                if "_" in j:
                    j = j.replace("_"," ")# remove _ for multiword expression
                    result.append(j)
                else:
                    result.append(j)
    
    return result

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str: 
    #predict the possible synonym with the highest occurence
    lemma = context.lemma
    pos = context.pos
    all_sets2 = wn.synsets(lemma, pos=pos)
    result = []
    occ = defaultdict(int)
    for i in all_sets2:
        for j in i.lemmas():
            name = j.name()
            if name not in result and name !=lemma:
                if "_" in name:
                    name = name.replace("_"," ")
                count = j.count()
                if name in occ:
                    occ[name]=occ[name]+count
                else:
                    occ[name]=count

    return max(occ, key = occ.get) # replace for part 2


# supply function for wn_simple_lesk_predictor
# count the overlap between definition of synset and context of target words
def overlap(context, definition):
    overlap = []
    stop_words = set(stopwords.words('english'))
    for target in context:
        if target in definition:
            if target not in overlap:
                if target not in stop_words:
                    overlap.append(target)
    return len(overlap) # number of overlap
# supply function for part 3
def frequency_target(synset, context):
    #count the frequency of target words and return the dicectory 
    lemma = context.lemma
    result = []
    frequency = 0
    occ = defaultdict(int)
    for j in synset.lemmas():
        name = j.name()
        if name not in result and name !=lemma:
            if "_" in name:
                name = name.replace("_"," ")
            count = j.count()
            if name in occ:
                occ[name]=occ[name]+count
            else:
                occ[name]=count
        elif name == lemma:
            c = j.count()
            frequency += c
    return occ, frequency
    

def wn_simple_lesk_predictor(context : Context) -> str:#simple Lesk Algorithm
    lemma = context.lemma
    pos = context.pos
    all_conwords = set(context.left_context + context.right_context)#possible synsets
    synset_count = defaultdict()
    lemmas = wn.lemmas(lemma, pos)# findthe synonyms of synset
    for i in lemmas:
        synset = i.synset()
        definition = tokenize(synset.definition())#definitions from synset

        #I got lost in this algorithm. I feel I need to first find the overlap and the frequency of the target 
        #words, but i don't know what i should do next.
    overlap_set = overlap(all_conwords, definition)
    count_dic, val_frequency = frequency_target(synset, context)
    final_result = 0
    final_count = 0
    while final_result !=0 and final_count !=0:
        if len(count_dic) !=0 :
            final_result = max(count_dic, key= count_dic.get)
            final_count = count_dic[final_result]
    return max(final_count, key = synset_count.get)#replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        #obtain a set of possible synonyms; return the synonym most similar to target word
        lemma = context.lemma
        pos = context.pos
        all_sets3 = wn.synsets(lemma, pos=pos)
        cans = get_candidates(lemma, pos)#set of possible synonyms from WordNet
        best= defaultdict()
        for i in cans:
            #if i in self.model.get_vector:
            try:
                best[i]= self.model.similarity(i,lemma)
            except:
                continue
        best_dict = max(best, key=best.get)
        return best_dict# replace for part 4


class BertPredictor(object):

    def __init__(self, filename): 
        
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        #feed the context sentence into BERT return a vector of length V and score for each word
        lemma = context.lemma
        pos = context.pos
        candidates = get_candidates(lemma, pos)
        all_context = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        index_mask = len(context.left_context)+1
        
        #I am not sure if what I am doing is correct, but I use all the functions from given instructions
        #input_toks = self.tokenizer.encode(all_context)
        input = self.tokenizer.convert_tokens_to_ids(all_context)
        input_mat = np.array(input).reshape((1,-1)) #get a 1*len(input) matrix reference from instruction
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][index_mask])[::-1]
        result= self.tokenizer.convert_ids_to_tokens(best_words)

        for i in result:
            if i in candidates:
                return i


         # replace for part 5
   

#part 6: personalized implement for lexical substitution task.

class BestWord(object):
    
    def __init__(self, filename): 
        self.model_gen = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True) 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    
    
    
    def BestPredict(self, context : Context):
        lemma = context.lemma
        pos = context.pos
        candidates = get_candidates(lemma, pos)
        all_context = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        index_mask = len(context.left_context)+1
        
        #similar to part 5; creates a DlstilBERT model
        input = self.tokenizer.convert_tokens_to_ids(all_context)
        input_mat = np.array(input).reshape((1,-1)) #get a 1*len(input) matrix reference from instruction
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][index_mask])[::-1]
        result= self.tokenizer.convert_ids_to_tokens(best_words)
        
        count=defaultdict()
        for i in candidates:
            if i in result:
                diff = self.model_gen.similarity(i, lemma)
                count[i]= -result.index(i)+20*diff #increase the index result
        
        return max(count, key=lambda x: count[x])
        

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = BestWord(W2VMODEL_FILENAME)
    #predictor = BertPredictor(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        print(context)  # useful for debugging
        prediction = predictor.BestPredict(context) 
        #prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    #print(get_candidates('slow','a'))
