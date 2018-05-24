# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:37:31 2018

@author: sanjana rajagopala
Corpus Statistics and Python Programming

"""
#Import the required packages
import nltk
import re
from nltk.corpus import PlaintextCorpusReader as pt
from nltk import FreqDist
from nltk.collocations import *
from operator import itemgetter


#Read the required text files from the path

#NOTE - Might require changing the path of the folder to ensure the corrent folder is read
pt_corpus = pt('C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_1','.*\.txt')
txt_files = pt_corpus.fileids()

#Read the part1 union addresses from the state_union_part1.txt file
state_union_text1 = pt_corpus.raw(txt_files[0])

#Read the part2 union addresses from the state_union_part2.txt file
state_union_text2 = pt_corpus.raw(txt_files[1])



#Check and explore the length of the text1
print("The length of the first state union address raw text {0}".format(len(state_union_text1)))
print("\n\n\n")
print("The length of the second state union address raw text {0}".format(len(state_union_text2)))
print(state_union_text1[:20])
print(state_union_text2[:20])


#Tokenize the first raw text
state_union_tokens1 = nltk.word_tokenize(state_union_text1)

#Tokenize the second raw text
state_union_tokens2 = nltk.word_tokenize(state_union_text2)

#Checking the tokens
state_union_tokens1[:20]
state_union_tokens2[:20]


#Processing the tokens before computing the frequency distribution table

#Step 1 - Conversion to lower case so that the occurrence of the token is considered as the same irrespective of the capitalization

state_union_tokens1 = [w.lower() for w in state_union_tokens1]
state_union_tokens2 = [w.lower() for w in state_union_tokens2]

#Step 2 - Removal of punctuations and numbers so that only sensible words are included 
#Define the function to filter 
#1. the numeric characters and special characters 
#2. to retain words with a hyphen in between 

def matchingNonAlpha(w):
    word_pattern = re.compile('^[a-z]+[-]*[a-z]+$')
    if(w.isnumeric() or not(word_pattern.match(w))):
        return True
    else:
        return False

#The modified list of tokens
state_revisedwords_1 = [w for w in state_union_tokens1 if(not matchingNonAlpha(w))]
state_revisedwords_2= [w for w in state_union_tokens2 if not(matchingNonAlpha(w))]


#Step 3 - Removal of the stop words because finding the most frequent words other than stop words better insights than with them

stopwords = nltk.corpus.stopwords.words('english')

state_revisedwords_1 = [w for w in state_revisedwords_1 if w not in stopwords]
state_revisedwords_2 = [w for w in state_revisedwords_2 if w not in stopwords]

#Step 4 - Must include lemmatization so that words such as interest and interests are not counted separately and formed as keys in frequency table
wnl = nltk.WordNetLemmatizer()
state_revised_lemma = [wnl.lemmatize(t) for t in state_revisedwords_1]
state_revised_lemma_2 = [wnl.lemmatize(t) for t in state_revisedwords_2]


#Checking the list of tokens after the above Preprocessing Steps
print(state_revised_lemma[:20])

print(state_revised_lemma_2[:20])


#Generating the Frequency Distribution 
freq_dist_1 = FreqDist(state_revised_lemma)
freq_dist_2 = FreqDist(state_revised_lemma_2)


#Finding the list of top 50 frequently occuring words normalized by the length of document, using the concept of percentage
freqWords = []
for key in freq_dist_1:
    freqWords.append((key,freq_dist_1.freq(key)*100))
    
freqWords_2 = []
for key in freq_dist_2:
    freqWords_2.append((key,freq_dist_2.freq(key)*100))
    
#Sort the list in descending order 
freqWords = sorted(freqWords, key = itemgetter(1),reverse=True)
freqWords_2 = sorted(freqWords_2, key = itemgetter(1),reverse=True)

#Print the words
print("\n\n\n")
print("*************The first state union adresses text - The top 50 by Frequency ( Normalized by length of the document) ************")
print(freqWords[:50])

print("\n\n\n")
print("*************The second state union adresses text -  The top 50 by Frequency ( Normalized by length of the document) ************")
print(freqWords_2[:50])

#Defining the collocation finder from the previous first set of lower case tokens

#Step 1- : Lemmatization
state_lemma_1 = [wnl.lemmatize(t) for t in state_union_tokens1]
state_lemma_2 = [wnl.lemmatize(t) for t in state_union_tokens2]


state_finder_1 = BigramCollocationFinder.from_words(state_lemma_1)
state_finder_2 = BigramCollocationFinder.from_words(state_lemma_2)


#Processing before finding the bigrams

#Step 1 - Removing the numeric and special characters

state_finder_1.apply_word_filter(matchingNonAlpha)

state_finder_2.apply_word_filter(matchingNonAlpha)

#Step 2 - Removing the bigrams having both the words of length <= 2

state_finder_1.apply_ngram_filter(lambda w1,w2: len(w1)<3 and len(w2) < 3)
state_finder_2.apply_ngram_filter(lambda w1,w2: len(w1)<3 and len(w2)<3)

#Step 3 - Remove only if both the words in the bigram are stop words 

state_finder_1.apply_ngram_filter(lambda w1,w2: w1 in stopwords and w2 in stopwords)
state_finder_2.apply_ngram_filter(lambda w1,w2:w1 in stopwords and w2 in stopwords)

#Finding the top 50 bigrams by frequency 
bigram_measures = nltk.collocations.BigramAssocMeasures()
score_raw_1 = state_finder_1.score_ngrams(bigram_measures.raw_freq)


print("\n\n\n")
print("***********The first state union adresses text - The top 50 bigrams by Frequency**********")
print(score_raw_1[:50])


print("\n\n\n")
print("***********The second state union adresses text - The top 50 bigrams by Frequency**********")
score_raw_2 = state_finder_2.score_ngrams(bigram_measures.raw_freq)
print(score_raw_2[:50])

#Step 5 - Filtering with the minimum frequency of 5
state_finder_1.apply_freq_filter(5)
state_finder_2.apply_freq_filter(5)


#Listing the top 50 bigrams by Mutual Information scores
print("\n\n\n")
print("***********The first state union adresses text - The top 50 bigrams by Mutual Information Scores**********")
score_pmi_1 = state_finder_1.score_ngrams(bigram_measures.pmi)
print(score_pmi_1[:50])



print("\n\n\n")
print("***********The second state union adresses text - The top 50 bigrams by Mutual Information Scores**********")
score_pmi_2 = state_finder_2.score_ngrams(bigram_measures.pmi)
print(score_pmi_2[:50])
