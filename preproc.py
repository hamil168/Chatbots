# -*- coding: utf-8 -*-
"""

Data preprocessing steps for Cornell Movie Script
Chatbot

using movie_conversations.text and
movie_lines.txt from the Cornell Movie Script Database


Created on Sat Jul 14 14:16:00 2018

@author: Ben Hamilton

#########
"""


# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

# - DATA PREPROCESSING ##########

# Importing the dataset (do this in the ipynb)
# lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
# conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


def id_to_line(lines):
    #create a dictionary mapping ids to lines
    # iterate through each line, split into different elements, get key, get value
    id2line = {}

    for line in lines:
      _line = line.split(' +++$+++ ')

      if len(_line) == 5:
        id2line[_line[0]] = _line[4]

    return id2line


def get_conversations_ids(conversations):

    # create a list of the conversations
    conversations_ids = []

    # The last row of data set is empty, so skip it
    for conversation in conversations[:-1]:

      # Split and remove brackets
      # Remove single quote
      # Remove spaces
      _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")

      # Append as list by spliting on commas
      conversations_ids.append(_conversation.split(","))

    return conversations_ids

def get_questions_and_answers(conversations_ids,id2line):
    # return unclean questions and answers
    # using the rule that every line that is responded to is a 'question'
    # and every line that is a response is the corresponding 'answer'
    # So it is expected that some lines appear on both lists, but not in
    # parallel to themselves.
    # Getting separately the questions and the answers
    questions = []
    answers = []

    for conversation in conversations_ids:

      for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

    return questionswords2int, answerswords2int, tokens


# first cleaning of the texts
def clean_text(text):
  text = text.lower()
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"that's", "that is", text)
  text = re.sub(r"what's", "what is", text)
  text = re.sub(r"where's", "where is", text)

  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "can not", text)
  text = re.sub(r"don't", "do not", text)

  text = re.sub(r"\'re", " are", text)

  text = re.sub(r"[-()\'#/@;:<>{}'\+\=\-\|.?,\!]", "", text)

  return text

# Apply on all "questions" and "answers"





#############################################################


def word_to_counts(clean_questions, clean_answers):
# Creating a function that tallies the count of each unique words
    word2count = {}
    for question in clean_questions:
      for word in question.split():
        if word not in word2count:
          word2count[word] = 1
        else:
          word2count[word] += 1

    for answer in clean_answers:
      for word in answer.split():
        if word not in word2count:
          word2count[word] = 1
        else:
          word2count[word] += 1

    return word2count

#############################################################


# Create 2 dictionaries that map questions words and answer words to integers
def map_questions_and_answers_to_integers(word2count):

    # set threshold for significance in word count
    threshold = 20

    questionswords2int = {}
    word_number = 0

    for word, count in word2count.items():
        if count >= threshold:
          questionswords2int[word] = word_number
          word_number += 1


    answerswords2int = {}
    word_number = 0

    for word, count in word2count.items():
        if count >= threshold:
          answerswords2int[word] = word_number
          word_number += 1

    # Adding the last tokens to these two dictionaries
    # create entries for tokens for questionswords2int and answerswords2uint

    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
      questionswords2int[token] = len(questionswords2int) + 1
      answerswords2int[token] = len(answerswords2int) + 1

    return questionswords2int, answerswords2int, tokens

#############################################################

def map_invert_answers_to_ints(answerswords2int):
    # Create dictionary that maps integers back to the answers
    # invert answersword2int using dict comprehension and .items

    answersints2word = {word_int: word for word, word_int in answerswords2int.items()}

#############################################################

def word_into_int(cleaned_text, word_dict):
    # turn a single cleaned text string into a list of
    # corresponding numbers
    word_into_ints_array = []

    for line in cleaned_text:
        ints = []

        for word in line.split():
            if word in word_dict:
                ints.append(word_dict[word])
            else:
                ints.append(word_dict['<OUT>'])

        word_into_ints_array.append(ints)

    return word_into_ints_array

#############################################################


def preproc_steps(lines, conversations):

    id2line = id_to_line(lines)

    conversations_ids = get_conversations_ids(conversations)

    questions, answers = get_questions_and_answers(conversations_ids,id2line)

    clean_questions = [clean_text(question) for question in questions]
    clean_answers = [clean_text(answer) for answer in answers]

    word2count = word_to_counts(clean_questions, clean_answers)

    questionswords2int, answerswords2int, tokens = map_questions_and_answers_to_integers(word2count)

    answersints2words = map_invert_answers_to_ints(answerswords2int)

    # Conccatenate <EOS> to every cleaned answer
    # needed for seq2seq model

    for i in range(len(clean_answers)):
      clean_answers[i] += ' <EOS>'

    # Translating cleaned questions into integers using
    # replace alal words filtered out by token with <OUT>

    questions_into_int = word_into_int(clean_questions, questionswords2int)

    answers_into_int = word_into_int(clean_answers, questionswords2int)

    """for question in clean_questions:
      ints = []

      # translate question into integers
      for word in question.split():

        if word in questionswords2int:
          ints.append(questionswords2int[word])
        else:
          ints.append(questionswords2int['<OUT>'])


      questions_into_int.append(ints)

    answers_into_int = []

    for answer in clean_answers:
      ints = []

      # translate answer into integers
      for word in answer.split():

        if word in answerswords2int:
          ints.append(answerswords2int[word])
        else:
          ints.append(answerswords2int['<OUT>'])"""

    # Sort questions by length of questions to speed up training
    # Reduces amount of padding during training

    sorted_clean_questions = []
    sorted_clean_answers = []

    # limit input to short sentences
    MAX_SENTENCE_LENGTH = 25

    # loop over possible lengths of questions
    for length in range(1, MAX_SENTENCE_LENGTH + 1):

      # use enumerate to loop 2 elements: index of question and question as list of ints
      for i in enumerate(questions_into_int):
        if i[0] in range(0,10):
            print(i)
        # if length of current question is equal to length we are checking...
        # append it to the sorted list by catching via the enumerated index
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int)#[i[0]])

            # snips answer length at 25, else goes to 500+ and OOM's the lstm train
            # keeps answer well aligned and truncated:
            curr_answer = answers_into_int[i[0]]
            trunc_answer = curr_answer[0:min(len(curr_answer), MAX_SENTENCE_LENGTH)]
            sorted_clean_answers.append(trunc_answer)

    return id2line, conversations_ids, questions, answers, clean_questions, clean_answers, word2count, sorted_clean_questions, sorted_clean_answers
