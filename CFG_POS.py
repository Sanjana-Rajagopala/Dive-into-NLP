# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:14:59 2018

@author: Sanjana Rajagopala
Context-Free Grammar
"""

#Import the required packages
import nltk

#Define the set of sentences 
sent1 = "We had a nice party yesterday"
sent2 = "She came to visit me two days ago"
sent3 = "You may go now"
sent4 = "Their kids are not always naive"

#Perform POS tagging for the current sentences so that the grammar can be framed accordingly

#Tokenize the sentences 
sent1_tokens = [ sent for sent in nltk.sent_tokenize(sent1)]
sent2_tokens = [ sent for sent in nltk.sent_tokenize(sent2)]
sent3_tokens = [ sent for sent in nltk.sent_tokenize(sent3)]
sent4_tokens = [ sent for sent in nltk.sent_tokenize(sent4)]

#Tokenize to get the words in the sentences
words1_tokens = [nltk.word_tokenize(s) for s in sent1_tokens]
words2_tokens = [nltk.word_tokenize(s) for s in sent2_tokens]
words3_tokens = [nltk.word_tokenize(s) for s in sent3_tokens]
words4_tokens = [nltk.word_tokenize(s) for s in sent4_tokens]

#Do the POS tagging using the Stanford NLP POS tagger
tagged_sent1 = [nltk.pos_tag(w) for w in words1_tokens]
tagged_sent2 = [nltk.pos_tag(w) for w in words2_tokens]
tagged_sent3 = [nltk.pos_tag(w) for w in words3_tokens]
tagged_sent4 = [nltk.pos_tag(w) for w in words4_tokens]

#Printing the POS tags of the words in every sentence
print("Sentence-1")
print(tagged_sent1)
print("\nSentence-2")
print(tagged_sent2)
print("\nSentence-3")
print(tagged_sent3)
print("\nSentence-4")
print(tagged_sent4)

#Constructing the grammar for the above sentences

#Separate VBs because give example of  a phrase like "are to" must not occur in a valid sentence
#Keeping the dtermeinants diverse

#Define the Noun Phrases with Proper nouns, Determiners, Adjectives and Adverbs
#Define the adverb phrases with adverbs
#Define the adjective clause with adjectives
#Define the verb phrases with verbs, modals, to-infinitives and adverbs
#Define the specific words from the above corpus - nouns, verbs, adverbs, adjectives, modals etc.



cur_grammar = nltk.CFG.fromstring('''
    S -> NP VP
    NP -> PRP |PRP ADVP | PRP NN | Det ADP NN 
    ADP -> JJ NN | JJ
    ADVP -> CD NN RB | RB ADP | RB ADVP | RB
    VP -> VPS NP | VPS TO VP | MD VP | VBP ADVP | VPS ADVP
    VPS -> VB | VBD 
    VB -> "visit" | "go" 
    VBD -> "had" | "came"
    VBP -> "are"
    JJ -> "nice" | "naive"
    Det -> "a"
    NN -> "party" | "yesterday" | "days" | "kids"
    PRP -> "We" | "She" | "me" | "You" | "Their"
    TO -> "to"
    CD -> "two"
    RB -> "ago" | "now" | "not" | "always"   
    MD -> "may"
    
'''
)

#Define the parser with the above grammar
rd_parser = nltk.RecursiveDescentParser(cur_grammar)

print(sent1)
sent1_list = sent1.split()
print(sent1_list)
for tree in rd_parser.parse(sent1_list):
    print(tree)

print(sent2)    
sent2_list = sent2.split()
print(sent2_list)
for tree in rd_parser.parse(sent2_list):
    print(tree)
    
print(sent3)    
sent3_list = sent3.split()
print(sent3_list)
for tree in rd_parser.parse(sent3_list):
    print(tree)

print(sent4)    
sent4_list = sent4.split()
print(sent4_list)
for tree in rd_parser.parse(sent4_list):
    print(tree)

#The three other sentences that can be parsed by the above grammar

my_sent1 = "She had to go now"
print(my_sent1)
my_sentlist1= my_sent1.split()
print(my_sentlist1)
for t in rd_parser.parse(my_sentlist1):
    print(t)
    
my_sent2 = "You are always nice"
print("\n"+my_sent2)
my_sentlist2= my_sent2.split()
print(my_sentlist2)
for t in rd_parser.parse(my_sentlist2):
    print(t)
    
my_sent3 = "We may visit a nice party"
print("\n"+my_sent3)
my_sentlist3= my_sent3.split()
print(my_sentlist3)
for t in rd_parser.parse(my_sentlist3):
    print(t)


#The sentence that does not make sense
no_sense_sent = "She visit me always"
no_sense_list= no_sense_sent.split()
print("Sentence with no sense\n")
print(no_sense_sent+"\n")
print(no_sense_list)
for t in rd_parser.parse(no_sense_list):
    print(t)
    
#Defining the probablistic CFG

#Take the above CFG and assign probablities for each rule depending upon their usage in the given corpus of 4 sentences

prob_grammar = nltk.PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> PRP [0.5]
    NP -> PRP ADVP [0.17]
    NP -> PRP NN [0.16]
    NP -> Det ADP NN [0.17]
    ADP -> JJ NN [0.5]
    ADP -> JJ [0.5]
    ADVP -> CD NN RB [0.25]
    ADVP -> RB ADP [0.25]
    ADVP -> RB ADVP [0.25]
    ADVP -> RB [0.25]
    VP -> VPS NP [0.34]
    VP -> VPS TO VP [0.17]
    VP -> MD VP [0.17]
    VP -> VPS ADVP [0.16]
    VP -> VBP ADVP [0.16]
    VPS -> VB [0.5] | VBD [0.5] 
    VB -> "visit" [0.5] | "go"  [0.5]
    VBD -> "had" [0.5] | "came" [0.5]
    VBP -> "are" [1.0]
    JJ -> "nice" [0.5] | "naive" [0.5]
    Det -> "a" [1.0] 
    NN -> "party"[0.25] | "yesterday" [0.25] | "days" [0.25] | "kids" [0.25]
    PRP -> "We" [0.2] | "She" [0.2] | "me" [0.2] | "You" [0.2] | "Their" [0.2]
    TO -> "to" [1.0]
    CD -> "two" [1.0]
    RB -> "ago" [0.25] | "now" [0.25] | "not" [0.25] | "always" [0.25]   
    MD -> "may" [1.0]
    
""")

#Define the Viterbi parser for the above PCFG
viterbi_parser = nltk.ViterbiParser(prob_grammar)

#Parse the given 4 sentence using the PCFG grammar

print(sent1)
print(sent1_list)
for tree in viterbi_parser.parse(sent1_list):
    print(tree)
    
print(sent2)    
print(sent2_list)
for tree in viterbi_parser.parse(sent2_list):
    print(tree)
    
print(sent3)    
print(sent3_list)
for tree in viterbi_parser.parse(sent3_list):
    print(tree)
    
print(sent4)    
print(sent4_list)
for tree in viterbi_parser.parse(sent4_list):
    print(tree)
    
#Parsing the other 3 sentences of my own using the PCFG
print(my_sent1)
print(my_sentlist1)
for t in viterbi_parser.parse(my_sentlist1):
    print(t)

print(my_sent2)
print(my_sentlist2)
for t in viterbi_parser.parse(my_sentlist2):
    print(t)
    
print(my_sent3)
print(my_sentlist3)
for t in viterbi_parser.parse(my_sentlist3):
    print(t)
    
#Parsing the sentence that does not make sense but parsed by the grammar using PCFG

print(no_sense_sent)
print(no_sense_list)
for t in viterbi_parser.parse(no_sense_list):
    print(t)
