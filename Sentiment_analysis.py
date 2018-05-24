#@Author Sanjana Rajagopala
#Sentiment Analysis
# coding: utf-8

# In[30]:


#Import all the required packages
import nltk

from nltk.tokenize import sent_tokenize
from nltk.corpus import sentence_polarity
import random

import collections
from nltk.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


# In[2]:


#####Extraction of the reviewTexts from the file


#Read the file content line by line 

inputPath = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/clothing_shoes_jewelry.txt"
inputTextData = open(inputPath).readlines()



#Maintain the list of reviewText sentences
reviewTextList = []
reviewYearList = []

#Use the regular Expressions to extract review Text content
pattern_1 = r'''(?x)
    reviewText[:](.+)
    '''

#Retrieve the year of the reviews
year_pattern = r''' (?x)
    reviewTime[:](.+)
'''
#Apply the regular expression 
#Strip off the newline character 

#Sentence tokenize each review text


#Taking the year 2013 and adding reviews of only that year
for line in inputTextData:
    
    sentence_list = []
    reviewyear_tokens = nltk.regexp_tokenize(line, year_pattern)
    reviewtext_tokens = nltk.regexp_tokenize(line, pattern_1)
        
    if(len(reviewtext_tokens)>0):
        current_review = reviewtext_tokens[0].strip("\n") 
        reviewTextList.append(current_review)
    if(len(reviewyear_tokens)>0):
        current_year = reviewyear_tokens[0].split(',')[1].strip('\n').strip(' ')
        reviewYearList.append(current_year)

        
#Taking only the year 2013 reviews
final_reviews = []
for i in range(0, len(reviewYearList)):
    if(reviewYearList[i]=='2013'):
        final_reviews.append(reviewTextList[i])
        
            
    
    


# In[3]:



#The final list of reviews contains 128518
print("The total number of reviews",len(final_reviews))

#Retreiving 10,000 reviews from this list
final_reviews = final_reviews[:10000]

print("The final number of reviews used for processing", len(final_reviews))


# In[23]:


final_reviews[:5]


# In[4]:


final_sent_list = []
#Obtain the sentences from the reviews list
for review in final_reviews:
    curr_sent_list = nltk.sent_tokenize(review)
    for sent in curr_sent_list:
        final_sent_list.append(sent)


# In[ ]:


final_sent_list[:5]


# In[5]:


#The number of sentences in the final reviews list
print("The total number of sentences from the reviews ", len(final_sent_list))


# In[6]:


###Pre processing the words in the review Text 


#1. Converting tolower case - As the focus is on the used word we can convert all words  into lower cases
#2. Removing the words that are only punctuations meaning just .,;!?

#Regular expression for punctuations
pattern_2 = r''' (?x)
    [][.,;"'?!():-_%']    
'''
temp_textList = []
for sent in final_sent_list:
   
    #Convert the sentence into lower case sentence
    sent = sent.lower()
    
    #Tokenize and obtain the words
    words = nltk.word_tokenize(sent)
    
    #If the sentence is a single punctuation mark without any other words then remove it to avoid processing in further analysis steps
    if(len(words)==1):
        if(len(nltk.regexp_tokenize(words[0], pattern_2))<0):
            temp_textList.append(sent)
    else:
        temp_textList.append(sent)
            


# In[7]:


len(temp_textList)


# In[8]:


#Copy back the updated list of processed sentences
final_sent_list = temp_textList


# In[9]:


#Check the length of review list
len(final_sent_list)


# In[10]:


stopwords = nltk.corpus.stopwords.words('english')


# In[11]:


##### Collect the sentences from the sentence_polarity corpus
bag_of_sents = sentence_polarity.sents()

#Obtain the sentences with their labels
sents_with_labels = [(sent, label) for label in sentence_polarity.categories()
                     for sent in sentence_polarity.sents(categories=label)]


#Shuffle the sentences
random.shuffle(sents_with_labels)

#Obtain the most frequent words from these sentences 
#Retreiving the most frequent 2000 words

all_sents = [word for (sent, label) in sents_with_labels for word in sent]
all_sents_freq = nltk.FreqDist(all_sents)
most_freq_words = all_sents_freq.most_common(2000)
freq_word_features = [word for (word,freq) in most_freq_words]


# In[12]:


#Define the function for word features
def document_features(sentence, words):
    unique_words = set(sentence)
    features = {}
    for word in words:
        features['contains({})'.format(word)] = (word in unique_words)
    return features


# In[13]:


#Obtaining the feature set for the sentences 
feature_set_1 = [(document_features(sent, freq_word_features), label) for (sent, label) in sents_with_labels]


# In[14]:


#Split the feature set data into training and test data
train_feature_set_1, test_feature_set_1 = feature_set_1[1000:], feature_set_1[:1000]

#Apply NB Classification on the training feature set
classifier_1 = nltk.NaiveBayesClassifier.train(train_feature_set_1)


# In[15]:


#Determine the Accuracy
print("The baseline accuracy of the first word features",nltk.classify.accuracy(classifier_1, test_feature_set_1))


# In[16]:


#Function to display all the measures for a given classifier
#@input: actual_set - the test features set
#@input: input_classifier -the classifier from which measures need to be obtained

def display_accuracy_measures(actual_set, input_classifier):
    reference_sets = collections.defaultdict(set)
    predicted_sets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(actual_set):
        reference_sets[label].add(i)
        predicted = input_classifier.classify(feats)
        predicted_sets[predicted].add(i)
    print('Positive precision:', precision(reference_sets['pos'], predicted_sets['pos']))
    print('Positive recall:', recall(reference_sets['pos'], predicted_sets['pos']))
    print('Positive F-measure:', f_measure(reference_sets['pos'], predicted_sets['pos']))
    print('Negative precision:', precision(reference_sets['neg'], predicted_sets['neg']))
    print ('Negative recall:', recall(reference_sets['neg'], predicted_sets['neg']))
    print ('Negative F-measure:', f_measure(reference_sets['neg'], predicted_sets['neg'])) 



# In[43]:


#Create the positive and negative lists
pos_sentence_list_1 = []
neg_sentence_list_1 = []

for sent in final_sent_list:
    if(classifier_1.classify(document_features(nltk.word_tokenize(sent),freq_word_features)) == 'pos'):
        pos_sentence_list_1.append(sent)
    elif(classifier_1.classify(document_features(nltk.word_tokenize(sent),freq_word_features)) == 'neg'):
        neg_sentence_list_1.append(sent)

    


# In[24]:


#Determine the Accuracy
print("The baseline accuracy of the first word features",nltk.classify.accuracy(classifier_1, test_feature_set_1))


# In[17]:


#Display the other measures
display_accuracy_measures(feature_set_1, classifier_1)


# In[44]:


#Printing the number of positive ann negative reviews
print("The number of positive reviews ",len(pos_sentence_list_1))
print("The number of negative reviews ",len(neg_sentence_list_1))


# In[111]:


#Define a function to write the sentences into positive and negative files
def writeIntoFile(file_path, sent_list):
    filepointer = open(file_path, 'w')
    for sent in sent_list:
        filepointer.write(sent)
        filepointer.write("\n")
  
    filepointer.close()


# In[ ]:


path_1 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_1.txt"
path_2 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_1.txt"

# write the returned results into the file
writeIntoFile(path_1, pos_sentence_list_1)
writeIntoFile(path_2, neg_sentence_list_1)


# In[18]:


#Feature-2 - Subjectivity Count Features
def readSubjectivity(path):
    flexicon = open(path, 'r')
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

SLpath = "C:/Users/sanja/Desktop/CoursesSpring18/IST664/Week_10/subjclueslen1-HLTEMNLP05.tff"
SL = readSubjectivity(SLpath)


# In[19]:


#Define the Subjectivity Count word features
def SL_features(sent, word_features, SL):
    sent_words = set(sent)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in sent_words)
        # count variables for the 4 classes of subjectivity
        weakPos = 0
        strongPos = 0
        weakNeg = 0
        strongNeg = 0
        for word in sent_words:
            if word in SL:
                strength, posTag, isStemmed, polarity = SL[word]
                if strength == 'weaksubj' and polarity == 'positive':
                    weakPos += 1
                if strength == 'strongsubj' and polarity == 'positive':
                    strongPos += 1
                if strength == 'weaksubj' and polarity == 'negative':
                    weakNeg += 1
                if strength == 'strongsubj' and polarity == 'negative':
                    strongNeg += 1
                features['positivecount'] = weakPos + (2 * strongPos)
                features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features



# In[20]:


#Obtaining the feature set for the sentences 
feature_set_2 = [(SL_features(sent, freq_word_features,SL), label) for (sent, label) in sents_with_labels]


# In[21]:


#Split the feature set data into training and test data
train_feature_set_2, test_feature_set_2 = feature_set_2[2000:], feature_set_2[:2000]

#Apply decision tree Classification on the training feature set
classifier_2 = nltk.DecisionTreeClassifier.train(train_feature_set_2)


# In[79]:


#Determine the Accuracy
print("The accuracy after adding Subjectivity Lexicon features ",nltk.classify.accuracy(classifier_2, test_feature_set_2))


# In[25]:


#Create the positive and negative lists
pos_sentence_list_2 = []
neg_sentence_list_2 = []

for sent in final_sent_list:
    if(classifier_2.classify(SL_features(nltk.word_tokenize(sent),freq_word_features, SL)) == 'pos'):
        pos_sentence_list_2.append(sent)
    elif(classifier_2.classify(SL_features(nltk.word_tokenize(sent),freq_word_features, SL)) == 'neg'):
        neg_sentence_list_2.append(sent)

    


# In[22]:


#Display the other measures
display_accuracy_measures(feature_set_2, classifier_2)


# In[27]:


#Printing the number of positive and negative reviews
print("The number of positive reviews ",len(pos_sentence_list_2))
print("The number of negative reviews ",len(neg_sentence_list_2))


# In[82]:


path_3 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_2.txt"
path_4 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_2.txt"


#Write the returned results into the files

writeIntoFile(path_3, pos_sentence_list_2)
writeIntoFile(path_4, neg_sentence_list_2)


# In[28]:


pos_sentence_list_2[:10]


# In[29]:


neg_sentence_list_2[:5]


# In[15]:


##Feature - 3: Negation Features

#Define the negation words
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']


# In[16]:


len(final_sent_list)


# In[17]:


def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # Parse the document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features


# In[18]:


#Obtaining the feature set for the sentences 
feature_set_3 = [(NOT_features(sent, freq_word_features, negationwords), label) for (sent, label) in sents_with_labels]


# In[19]:


#Split the feature set data into training and test data
train_feature_set_3, test_feature_set_3 = feature_set_3[2000:], feature_set_3[:2000]

#Apply NB Classification on the training feature set
classifier_3 = nltk.NaiveBayesClassifier.train(train_feature_set_3)


# In[20]:


#Determine the Accuracy
print("The accuracy after adding Negation features ",nltk.classify.accuracy(classifier_3, test_feature_set_3))


# In[46]:


#Display the other measures
display_accuracy_measures(feature_set_3, classifier_3)


# In[21]:


#Create the positive and negative lists
pos_sentence_list_3 = []
neg_sentence_list_3 = []

for sent in final_sent_list:
    if(classifier_3.classify(NOT_features(nltk.word_tokenize(sent),freq_word_features, negationwords)) == 'pos'):
        pos_sentence_list_3.append(sent)
    elif(classifier_3.classify(NOT_features(nltk.word_tokenize(sent),freq_word_features,negationwords)) == 'neg'):
        neg_sentence_list_3.append(sent)

    


# In[24]:


path_5 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_3.txt"
path_6 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_3.txt"


#Write the returned results into the files

writeIntoFile(path_5, pos_sentence_list_3)
writeIntoFile(path_6, neg_sentence_list_3)


# In[ ]:


#Printing the number of positive and negative reviews
print("The number of positive reviews ",len(pos_sentence_list_3))
print("The number of negative reviews ",len(neg_sentence_list_3))


# In[25]:


#Feature 4 : Removing the stopwords


#Remove the negated stopwords from the original stopwords list
stopwords = nltk.corpus.stopwords.words('english')
new_stopwords_list = [word for word in stopwords if word not in negationwords]


#Obtain the most frequent words from these sentences 
#Retreiving the most frequent 2000 words

all_sents = [word for (sent, label) in sents_with_labels for word in sent]
all_sents_freq = nltk.FreqDist(all_sents)
most_freq_words = all_sents_freq.most_common(2000)
new_freq_word_features = [word for (word,freq) in most_freq_words]


# In[26]:


#Re-running all the classifiers with the new feature set

#Word Features
#Obtaining the feature set for the sentences 
new_feature_set_1 = [(document_features(sent, new_freq_word_features), label) for (sent, label) in sents_with_labels]

#Split the feature set data into training and test data
new_train_feature_set_1, new_test_feature_set_1 = new_feature_set_1[1000:], new_feature_set_1[:1000]

#Apply NB Classification on the training feature set
new_classifier_1 = nltk.NaiveBayesClassifier.train(new_train_feature_set_1)

#Determine the Accuracy
print("The new accuracy with word features after removing stopwords ",nltk.classify.accuracy(new_classifier_1, new_test_feature_set_1))


# In[27]:


#Create the positive and negative lists
pos_sentence_list_4 = []
neg_sentence_list_4 = []

for sent in final_sent_list:
    if(new_classifier_1.classify(document_features(nltk.word_tokenize(sent),new_freq_word_features)) == 'pos'):
        pos_sentence_list_4.append(sent)
    elif(new_classifier_1.classify(document_features(nltk.word_tokenize(sent),new_freq_word_features)) == 'neg'):
        neg_sentence_list_4.append(sent)


# In[48]:


#Display the other measures for the classifier
display_accuracy_measures(new_feature_set_1, new_classifier_1)


# In[49]:


path_7 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_4.txt"
path_8 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_4.txt"

# write the returned results into the file
writeIntoFile(path_7, pos_sentence_list_4)
writeIntoFile(path_8, neg_sentence_list_4)


# In[ ]:


#Printing the number of positive and negative reviews
print("The number of positive reviews ",len(pos_sentence_list_4))
print("The number of negative reviews ",len(neg_sentence_list_4))


# In[56]:


#Obtaining the feature set for the sentences 
new_feature_set_2 = [(SL_features(sent, new_freq_word_features, SL), label) for (sent, label) in sents_with_labels]

#Split the feature set data into training and test data
new_train_feature_set_2, new_test_feature_set_2 = new_feature_set_2[1000:], new_feature_set_2[:1000]

#Apply NB Classification on the training feature set
new_classifier_2 = nltk.DecisionTreeClassifier.train(new_train_feature_set_2)

#Determine the Accuracy
print("The new accuracy with Subjectivity Count features after removing stopwords ",nltk.classify.accuracy(new_classifier_2, new_test_feature_set_2))


# In[57]:


#Create the positive and negative lists
pos_sentence_list_5 = []
neg_sentence_list_5 = []

for sent in final_sent_list:
    if(new_classifier_2.classify(SL_features(nltk.word_tokenize(sent),new_freq_word_features, SL)) == 'pos'):
        pos_sentence_list_5.append(sent)
    elif(new_classifier_2.classify(SL_features(nltk.word_tokenize(sent),new_freq_word_features, SL)) == 'neg'):
        neg_sentence_list_5.append(sent)


# In[58]:


#Display the other measures for the classifier
display_accuracy_measures(new_feature_set_2, new_classifier_2)


# In[60]:


path_9 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_5.txt"
path_10 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_5.txt"

# write the returned results into the file
writeIntoFile(path_9, pos_sentence_list_5)
writeIntoFile(path_10, neg_sentence_list_5)


# In[ ]:


#Printing the number of positive and negative reviews
print("The number of positive reviews ",len(pos_sentence_list_5))
print("The number of negative reviews ",len(neg_sentence_list_5))


# In[61]:


#Obtaining the feature set for the sentences 
new_feature_set_3 = [(NOT_features(sent, new_freq_word_features, negationwords), label) for (sent, label) in sents_with_labels]

#Split the feature set data into training and test data
new_train_feature_set_3, new_test_feature_set_3 = new_feature_set_3[1000:], new_feature_set_3[:1000]

#Apply NB Classification on the training feature set
new_classifier_3 = nltk.NaiveBayesClassifier.train(new_train_feature_set_3)

#Determine the Accuracy
print("The new accuracy with Negation features after removing stopwords ",nltk.classify.accuracy(new_classifier_3, new_test_feature_set_3))


# In[62]:


#Create the positive and negative lists
pos_sentence_list_6 = []
neg_sentence_list_6 = []

for sent in final_sent_list:
    if(new_classifier_3.classify(NOT_features(nltk.word_tokenize(sent),new_freq_word_features, negationwords)) == 'pos'):
        pos_sentence_list_6.append(sent)
    elif(new_classifier_3.classify(NOT_features(nltk.word_tokenize(sent),new_freq_word_features, negationwords)) == 'neg'):
        neg_sentence_list_6.append(sent)


# In[65]:


#Display the other measures for the classifier
display_accuracy_measures(new_feature_set_3, new_classifier_3)


# In[64]:


path_11 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_6.txt"
path_12 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_6.txt"

# write the returned results into the file
writeIntoFile(path_11, pos_sentence_list_6)
writeIntoFile(path_12, neg_sentence_list_6)


# In[ ]:


#Printing the number of positive and negative reviews
print("The number of positive reviews ",len(pos_sentence_list_6))
print("The number of negative reviews ",len(neg_sentence_list_6))


# In[ ]:


review_Data =  [sent[0] for sent in sents_with_labels]


# In[75]:


reviews = []


# In[76]:


for sent in review_Data:
    reviews.append(" ".join(x for x in sent))


# In[ ]:


review_target = [sent[1] for sent in sents_with_labels]


# In[79]:



#My Experiment
#Use the count vectorizer with minimum document frequency as 3 
from sklearn.feature_extraction.text import CountVectorizer


count_vec = CountVectorizer(min_df=3, tokenizer=nltk.word_tokenize)         

sents_counts = count_vec.fit_transform(reviews)


# In[80]:


# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
sents_tfidf = tfidf_transformer.fit_transform(sents_counts)




# In[81]:


sents_counts.shape


# In[82]:


from sklearn.svm import SVC
# Split data into training and test sets
# from sklearn.cross_validation import train_test_split  

from sklearn.model_selection import train_test_split

docs_train, docs_test, y_train, y_test = train_test_split(
    sents_tfidf, review_target, test_size = 0.20, random_state = 12)


# In[83]:


clf = SVC().fit(docs_train, y_train)


# In[85]:


# Predicting the Test set results, find accuracy
import sklearn
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[86]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[103]:


#Take the product reviews 
reviews_new_counts = count_vec.transform(final_sent_list)
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)


# In[104]:


# Use classifier make a prediction
review_prediction = clf.predict(reviews_new_tfidf)


# In[109]:


# Add into a list based on category and write into a file
pos_sentence_list_7 = []
neg_sentence_list_7 = []
for review, category in zip(final_sent_list, review_prediction):
    if(category == 'pos'):
        pos_sentence_list_7.append(review)
    else:
        neg_sentence_list_7.append(review)


# In[112]:


path_13 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/pos_model_7.txt"
path_14 = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_4/neg_model_7.txt"

# write the returned results into the file
writeIntoFile(path_13, pos_sentence_list_7)
writeIntoFile(path_14, neg_sentence_list_7)

