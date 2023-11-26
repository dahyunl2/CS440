# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    p_words=[]
    n_words=[]
    pos_counter=Counter()
    neg_counter=Counter()
    for review, label in zip(train_set, train_labels):  
        if label==0:
            n_words+=review
            neg_counter.update([x for x in review])
        else:
            p_words+=review
            pos_counter.update([x for x in review])
    p_word_cnt=len(p_words)
    p_v=len(set(p_words))
    n_word_cnt=len(n_words)
    n_v=len(set(n_words))

    p_words_bi=[]
    n_words_bi=[]
    pos_counter_bi=Counter()
    neg_counter_bi=Counter()
    for review, label in zip(train_set, train_labels):  
        if label==0:
            r=[]
            first=review[0]
            for i in review:
                tup=(first, i)
                first=i
                r.append(tup)
                n_words_bi.append(tup)
            neg_counter_bi.update([x for x in r])
        else:
            r=[]
            first=review[0]
            for i in review:
                tup=(first, i)
                first=i
                r.append(tup)
                p_words_bi.append(tup)
            pos_counter_bi.update([x for x in r])
    p_word_cnt_bi=len(p_words_bi)
    p_v_bi=len(set(p_words_bi))
    n_word_cnt_bi=len(n_words_bi)
    n_v_bi=len(set(n_words_bi))


    for doc in tqdm(dev_set, disable=silently):
        p_product=math.log(pos_prior)
        n_product=math.log(1-pos_prior)
        first=doc[0]
        cnt=0
        for j in doc:
            p_product+=(1-bigram_lambda)*math.log((pos_counter[j]+unigram_laplace)/(p_word_cnt+unigram_laplace*(p_v+1)))
            n_product+=(1-bigram_lambda)*math.log((neg_counter[j]+unigram_laplace)/(n_word_cnt+unigram_laplace*(n_v+1)))
            if cnt>0:
                tup=(first,j)
                first=j
                p_product+=bigram_lambda*math.log((pos_counter_bi[tup]+bigram_laplace)/(p_word_cnt_bi+bigram_laplace*(p_v_bi+1)))
                n_product+=bigram_lambda*math.log((neg_counter_bi[tup]+bigram_laplace)/(n_word_cnt_bi+bigram_laplace*(n_v_bi+1)))
            else: 
                cnt+=1
            
        if p_product<=n_product:
            yhats.append(0)
        else:
            yhats.append(1)

    return yhats



