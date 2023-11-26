"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

# def viterbi_3(train, test):
#     '''
#     input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#             test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
#     output: list of sentences, each sentence is a list of (word,tag) pairs.
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''
#     return []

"""
Part 2: This is the simplest ver of viterbi that doesn't do anything spec for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log
from utils import START_TAG
import copy

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

def pattern(word):
    if len(word)>=5:
        if word[-4:]=='tion' or word[-5:]=='tions':
            return 'P_tion'
        if word[-4:]=='sion' or word[-5:]=='sions':
            return 'P_sion'
    
    if len(word)>=3:
        if word[-3:]=='ing':
            return 'P_ing'
        if word[-3:]=='est':
            return 'P_est'
        if word[-3:]=='ful':
            return 'P_ful'
        if word[-2:]=='er' or word[-3:]=='ers':
            return 'P_er'
        if word[-2:]=='or' or word[-3:]=='ors':
            return 'P_or'
    
    if len(word)>=2:
        if word[-2:]=='ly':
            return 'P_ly'
        if word[-2:]=="'s":
            return "P_'s"
        if word[-2:]=="'t":
            return "P_'t"
        if word[-2:]=='ed':
            return 'P_ed'
        if word[-2:]=='en':
            return 'P_en'
        if word[-2:]=='id':
            return 'P_id'
        
    if len(word)>=1:
        if word[-1:].isnumeric():
            return 'num'
        # if word[1:]=='$':
        #     return '$'
    return ''
        
    
def training(sentences):
    """
    Computes init tags, emis words and trans tag-to-tag probabilities
    :param sentences:
    :return: intit tag probs, emis words given tag probs, trans of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    
    # init_prob={}

    #trans_prob __________________________________________________________________________________
    trans_prob_count = {}
    for sentence in sentences:
        for word in range(len(sentence) - 1):
            if word==0:
                init_prob[sentence[word][1]]=1+init_prob.get(sentence[word][1], 0)
            if sentence[word][1] not in trans_prob_count:
                trans_prob_count[sentence[word][1]]={}
                trans_prob_count[sentence[word][1]][sentence[word + 1][1]]=1
            elif sentence[word + 1][1] not in trans_prob_count[sentence[word][1]]:
                trans_prob_count[sentence[word][1]][sentence[word + 1][1]]=1
            else:
                trans_prob_count[sentence[word][1]][sentence[word + 1][1]]+=1
    # print(trans_prob_count)
    for taga in trans_prob_count:
        for tagb in trans_prob_count[taga]:
            # print(len(trans_prob_count[taga]))
            trans_prob[taga][tagb]= (trans_prob_count[taga][tagb]) / (sum(trans_prob_count[taga].values()))
   
    for tag in init_prob:
        init_prob[tag]=init_prob[tag]/len(sentences)

    
    #emit_prob __________________________________________________________________________________
    words={}
    emit_prob_count = {}
    for sentence in sentences:
        for pair in sentence:
            if pair[0] not in words:
                words[pair[0]]=1
            else:
                words[pair[0]]+=1
            if pair[1] not in emit_prob_count:
                emit_prob_count[pair[1]] = {}
            if pair[0] not in emit_prob_count[pair[1]]:
                emit_prob_count[pair[1]][pair[0]] = 1
            else:
                emit_prob_count[pair[1]][pair[0]] += 1

    hapax_count={}
    emit_prob_count_copy=copy.deepcopy(emit_prob_count)
    for tag in emit_prob_count_copy:
        for word in emit_prob_count_copy[tag]:
            if words[word]== 1:
                pat=pattern(word)
                if pat!='':
                    if pat in emit_prob_count[tag]:
                        emit_prob_count[tag][pat] += 1e-5
                    else:
                        emit_prob_count[tag][pat] = 1e-5
                    # emit_prob_count[tag][word]-=0.5
                if tag not in hapax_count:
                    hapax_count[tag]=1
                else:
                    hapax_count[tag]+=1
    hapax_num=sum(hapax_count.values())
    for tag in hapax_count:
        hapax_count[tag]=hapax_count[tag]/hapax_num
    

    for tag in emit_prob_count:
        total = sum(emit_prob_count[tag].values())
        emit_prob[tag]={}
        if tag not in hapax_count:
                hapax=1/hapax_num
        else:
            hapax=hapax_count[tag]
        for word in emit_prob_count[tag]:
            emit_prob[tag][word]= (emit_prob_count[tag][word] + epsilon_for_pt*hapax) / (total + epsilon_for_pt*hapax * (len(emit_prob_count[tag]) + 1))
        emit_prob[tag]['unseen']= (epsilon_for_pt*hapax) / (total + epsilon_for_pt*hapax * (len(emit_prob_count[tag]) + 1))       
    
    # print(emit_prob['NOUN'])
    return init_prob, emit_prob, trans_prob 

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emis probabilities
    :param trans_prob: Trans probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    if i==0:
        for tag in prev_prob:
            # print(word)
            # print(emit_prob)
            if word in emit_prob[tag]:
                log_prob[tag]=log(emit_prob[tag][word])+(prev_prob[tag])
                
            else:
                pat=pattern(word)
                if pat in emit_prob[tag]:
                    log_prob[tag]=log(emit_prob[tag][pat])+(prev_prob[tag])
                else:
                    log_prob[tag]=log(emit_prob[tag]['unseen'])+(prev_prob[tag])
                
            predict_tag_seq[tag]=[tag]

    for tagb in prev_prob:
            max=-100000000000
            max_tag=''
            for taga in prev_prob:
                # print(prev_prob[taga])
                prob= (prev_prob[taga])
                if taga not in trans_prob:
                    prob+=log(epsilon_for_pt)
                    # prob+=0
                elif tagb not in trans_prob[taga]:
                    prob+=log(epsilon_for_pt)
                    # prob+=0
                else:
                    prob+=log(trans_prob[taga][tagb])
                if word in emit_prob[tagb]:
                    prob+=log(emit_prob[tagb][word])
                else:
                    pat=pattern(word)
                    if pat in emit_prob[tagb]:
                        prob+=log(emit_prob[tagb][pat])
                        # print(tagb, pat, emit_prob[tagb][pat])
                    else:
                        prob+=log(emit_prob[tagb]['unseen'])
                    
                # print(trans_prob[tagb][taga])
                # print(emit_prob[tagb][word])
                # else:
                #     prob= (prev_prob[taga])+log(trans_prob[tagb][taga])+log(emit_prob[tagb][word])
                if prob>max:
                    max=prob
                    max_tag=taga
            log_prob[tagb]=max
            
            # print('maxtag',max_tag)
            predict_tag_seq[tagb]=prev_predict_tag_seq[max_tag]+[tagb]
    
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 spec case.
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        max=-10000000000000
        max_tag=''
        for tag in log_prob:
            if log_prob[tag]>max:
                max=log_prob[tag]
                max_tag=tag
        max_sequence=predict_tag_seq[max_tag]

        predicts.append([])
        for i in range(len(max_sequence)):
            predicts[sen].append((sentence[i], max_sequence[i]))

    return predicts




