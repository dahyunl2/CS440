# naive_bayes.py
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=8.0, pos_prior=10000/13000, silently=False):
    print_values(laplace,pos_prior)
    yhats = []

    p_words=[]
    n_words=[]
    pos_counter=Counter()
    neg_counter=Counter()
    for review, label in zip(train_set, train_labels):  
        if label==0:
            n_words+=review
            neg_counter.update([x.lower() for x in review])
        else:
            p_words+=review
            pos_counter.update([x.lower() for x in review])
    p_word_cnt=len(p_words)
    p_v=len(set(p_words))
    n_word_cnt=len(n_words)
    n_v=len(set(n_words))

    for doc in tqdm(dev_set, disable=silently):
        p_product=math.log(pos_prior)
        n_product=math.log(1-pos_prior)
        for j in doc:
            p_product+=math.log((pos_counter[j.lower()]+laplace)/(p_word_cnt+laplace*(p_v+1)))
            n_product+=math.log((neg_counter[j.lower()]+laplace)/(n_word_cnt+laplace*(n_v+1)))
        
        if p_product<=n_product:
            yhats.append(0)
        else:
            yhats.append(1)
                
    return yhats