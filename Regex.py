# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:22:06 2018

@author: Sanjana Rajagopala
Regular Expressions

"""
#Import the required packages
import nltk
import re

#Define the patterns
#1. Extract the File name
#2. Extract the NSF Org
#3. Extract the Award Amount
#4. Extract the Abstract text

patterns = r''' (?x) 
   (?:File\s+:\s*(\w+))
   | NSF\s+Org\s+:\s+(\w+)
   | Total\sAmt\.\s+:\s+(\$\d+)
  
     '''
#To obtain the entire Abstract text wiht the Abstract word
abs_pattern = "Abstract\s*:(.*)"

#To strip off the additional characters such as //*** after the end of the paragraph in the Abstract text.
abs_pattern_2 = r''' (?x)
    Abstract\s*:(.*)[!\?\.]
    | Abstract\s*:([\w\s\.,;:]+)
    
    
    '''
#NOTE - Might require changing the path of the folder to ensure the corrent folder is read or written

#Create the output file to write the first set of results 
outputFile = open("C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_2/Output_2.txt","w")

#Create the output file to write the second set of results 
outputFile_2 = open("C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_2/Output_3.txt","w")

#Write the first initial heading sentence in the file 

initial_heading = ["Abstract_ID","Sentence_No","Sentence"]
intitial_sent = " | ".join(initial_heading)
outputFile_2.write(intitial_sent)
outputFile_2.write("\n")
outputFile_2.write("-----------------------------------------------------------")

#Import the required packages
from nltk.tokenize import sent_tokenize


#Add the abstract id \ text line in the first line
#Define the function that takes the entire file text and file name as the inputs. It transforms the input into required ersult format
def identifySent(fileText, fileName):
    sentence_string = ""
    sent_count = 1
    sent_total = "Number of sentences : "
    #Use sentence tokenizer to obtain the sentences in the input file text
    sentence_list = sent_tokenize(fileText)
    
    outputFile_2.write("\n")
    #Create the line with filename, sentence number and the sentence
    for sent in sentence_list:
        sentence_string += fileName
        sentence_string += "|" +(str(sent_count))
        sentence_string += "|" +sent
        
        outputFile_2.write(sentence_string)
        outputFile_2.write("\n")
        sent_count +=1
        sentence_string = ""
        
   
    outputFile_2.write("\n")
    outputFile_2.write(sent_total + str(len(sentence_list)))
    outputFile_2.write("\n")
    outputFile_2.write("-----------------------------------------------------------")
    
    listOfSents.append(len(sentence_list))
    
#For Analysis of results
listOfFiles = []
listOfAmounts = []
listOfOrgs = []
listOfSents = []

#Define the function that does the same processing on each of the files
def processFiles(lines):
    abstract_text = ""
    isAbs = False
    temp_file = ""
    temp_org = ""
    temp_amnt = ""

    #First, check for Abstract text, extract it and maintain in separate string
    for line in lines:
        
        if(len(nltk.regexp_tokenize(line, abs_pattern))>0 or isAbs == True):
            isAbs = True
            abstract_text += line
    #Add the tokenized terms into the separate list
        else:
            token_list = nltk.regexp_tokenize(line, patterns)
            if(len(token_list)>0):
                if(token_list[0][0]!=''):
                    temp_file = token_list[0][0]
                if(token_list[0][1]!=''):
                    temp_org = token_list[0][1]
                if(token_list[0][2]!=''):
                    temp_amnt = token_list[0][2]
                    
   
    #Return if the abstract text is empty
    if(abstract_text == ''):
        return ""
    
    
    
    #Check the matching tuple because some abstract text do not end with . but some end with ***//. 
    #To handle both the cases, this processing step is done 
    #Replace the newline and extra space characters with single space character.
    temp_abs = nltk.regexp_tokenize(abstract_text, abs_pattern_2)[0]
    if(temp_abs[0]!=''):
        final_abs_text = temp_abs[0]
    elif(temp_abs[1]!=''):
        final_abs_text = temp_abs[1]
    
   
    final_abs_text = final_abs_text.replace('\n',"")
    final_abs_text = re.sub("\s+",' ', final_abs_text)
    

    #Create the final string containing all the above terms with space as delimiter
    final_output_line = [temp_file,temp_org,temp_amnt,final_abs_text]
    final_output_line = " ".join(final_output_line)
    
  
    #Call the function that writes the sentence details into the output file 2B
    identifySent(final_abs_text, temp_file)
    
    listOfFiles.append(temp_file)
    listOfOrgs.append(temp_org)
    listOfAmounts.append(temp_amnt)
    
    #Return this modified string
    return final_output_line
    
import glob
#Read the text files from the NFS_abstracts folder
path = "C://Users/sanja/Desktop/CoursesSpring18/IST664/Assignments/Assignment_2/NSF_abstracts/*.txt"


#For each file perform the same processing steps and write the returned results into the file
for fl in glob.glob(path):
    lines = open(fl).readlines()
    outputFile.write(processFiles(lines))
    outputFile.write("\n")
    

#Close the output results file
outputFile.close()
outputFile_2.close()

##################################For Analysis of Results###########################################################

#Getting teh max and min amounts abstracts
listOfAmounts = list(map(lambda x : x[1:], listOfAmounts))
listOfAmounts = list(map(int, listOfAmounts))
max_amnt = max(listOfAmounts)
min_amnt = min(listOfAmounts)

print(max_amnt)
print(min_amnt)

print(listOfAmounts.index(max_amnt))
print(listOfAmounts.index(min_amnt))
print(listOfFiles[364])
print(listOfFiles[68])
print(listOfOrgs[364])
print(listOfOrgs[68])

#Creating a data frame for the results
import pandas as pd 

df = pd.DataFrame({'files':listOfFiles, 'orgs': listOfOrgs, 'amnts': listOfAmounts, 'sents': listOfSents})
orgsGroup = df.groupby('orgs')
orgsGroup.apply(lambda x : max(x['amnts']))
sampl = orgsGroup.apply(lambda x : sum(x['amnts']))
print(sampl)
min(sampl)
max(sampl) #---> OCE 

sampl_2 = orgsGroup.apply(lambda x : len(x['amnts']))
print(sampl_2)
max(sampl_2) #---->DMS

sampl_3 = (orgsGroup.apply(lambda x : max(x['sents'])))
print(sampl_3)
max(sampl_3) #---->SES

#Approximately 72 abstracts with Not Available abstract texts

for i in df.index:
    if(df.loc[i,'amnts']==0):
        print(df.loc[i,'orgs'])
                
#Majorly INT org with 0 amounts
