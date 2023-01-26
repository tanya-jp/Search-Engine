!pip install hazm
!pip3 install parsivar
import json
import re
import math 
import numpy as np
from copy import deepcopy
from itertools import islice
from google.colab import drive
from hazm import utils
from parsivar import Tokenizer, Normalizer, FindStems


drive.mount('/content/drive')

f = open("/content/drive/MyDrive/IR_data_news_12k.json")
news_dataset = json.load(f)

def stemming(tokens, docID, mode):
    # mode=1 for stemming in news content, mode=2 for stemming in query for stemming in query
    stemmed_token_list = []
    my_stemmer = FindStems()
    for token in tokens: 
      stemmed_token = my_stemmer.convert_to_stem(token)
      stemmed_token = stemmed_token.split('&')
      if mode == 1:
        stemmed_token_list.append((stemmed_token[0], int(docID)))
      elif mode == 2:
        stemmed_token_list.append(stemmed_token[0])
    return stemmed_token_list


def preprocessing(news_dataset):
   tokenizer = Tokenizer()
   normalizer = Normalizer(statistical_space_correction=True)
   
   term_docID = []
   news_title_url = {}
  
   counter = 0
   for docID in news_dataset:
     if int(docID) % 1000 == 0:
       print(docID)
     content = news_dataset[docID]['content']
     url =  news_dataset[docID]['url']
     title =  news_dataset[docID]['title']
     news_title_url[int(docID)] = [title, url]
     # removing punctuations from content
     new_content = re.sub(r'[^\w\s]', '', content)
     # normalize new content
     normalized_content = normalizer.normalize(new_content)
     # getting the tokens(non-positional)
     tokenized_content = tokenizer.tokenize_words(normalized_content)
     # removing stopwords
     stopwords = utils.stopwords_list()

     for token in deepcopy(tokenized_content):
        if token in stopwords:
            tokenized_content.remove(token)

     # stemming 
     term_docID.extend(stemming(tokenized_content, docID, 1))
    
   return term_docID, news_title_url

term_docID, news_title_url = preprocessing(news_dataset)

# positions as int
def positional_indexing(term_docID):
  positional_index = {}
  cnt = 0

  pre_docID = 0
  position = 1

  for item in term_docID:
    token = item[0]
    docID = item[1]
    if pre_docID != docID:
      position = 1

    position_dic = {} 
    if token not in positional_index:     
      position_dic[docID] = 1
      positional_index[token] = [1, position_dic]

    else:
      value = positional_index[token]
      position_dic = value[1]

      if docID not in position_dic:  
        rep = value[0]
        rep += 1
        value[0] = rep
        position_dic[docID] = 0
        
      position_dic[docID] += 1
      

      positional_index[token] = value
    
    position += 1
    pre_docID = docID
    cnt += 1
    if cnt % 200000 == 0:
      print(cnt)

  # save the dictionary to check it
  try:
    positional_index_file = open('positional_index.txt', 'wt')
    positional_index_file.write(str(positional_index))
    positional_index_file.close()
  
  except:
    print("Unable to write to file")
  
  return positional_index

positional_index = positional_indexing(term_docID)


def take(n, iterable):
    # Return the first n items of the iterable as a list
    return list(islice(iterable, n))

# build champions list 
def build_champions_list(positional_index):
  champions_list = {}

  for term in positional_index:
    postings = positional_index[term][1]

    # finding the most relevant docs to term
    sorted_postings = {}
    for doc, term_freq in  sorted(postings.items(), key=lambda item: item[1], reverse=True):
      sorted_postings[doc] = term_freq

    # returns list of dictionaries
    most_relevant_docs_list = take(50, sorted_postings.items())

    candidate_list = {}
    for candidate_doc in most_relevant_docs_list:  
      candidate_list[candidate_doc[0]] = candidate_doc[1]
    
    champion_list = [positional_index[term][0], candidate_list]
    champions_list[term] = champion_list

  # save the dictionary to check it
  try:
    champions_list_file = open('champions_list.txt', 'wt')
    champions_list_file.write(str(positional_index))
    champions_list_file.close()
  
  except:
    print("Unable to write to file")

  return champions_list

champions_list = build_champions_list(positional_index)


def query_preprocessing(query_content):
  tokenizer = Tokenizer()
  normalizer = Normalizer(statistical_space_correction=True)

  preprocessed_query = []
  # removing punctuations from content
  no_punc_query_content = re.sub(r'[^\w\s]', '', query_content)
  # normalize new content
  normalized_query_content = normalizer.normalize(no_punc_query_content)
  # getting the tokens(non-positional)
  tokenized_query_content = tokenizer.tokenize_words(normalized_query_content)
  # remove stopwords
  stopwords = utils.stopwords_list()

  for token in deepcopy(tokenized_query_content):
    if token in stopwords:
        tokenized_query_content.remove(token)
  # stemming 
  preprocessed_query.extend(stemming(tokenized_query_content, -1, 2))

  return preprocessed_query

# calculating length of each document 
def calculate_doc_length(positional_index, doc_num):
  doc_length_array = np.zeros(doc_num)
  
  # each doc is a vector of tf.idf weights 

  for t in positional_index:   
    for d in positional_index[t][1]:  
      tf = 1 + math.log10(positional_index[t][1][d])
      idf = math.log10(doc_num / positional_index[t][0])            
      doc_length_array[int(d)] += (tf * idf)**2
      
  doc_length_array = doc_length_array ** 0.5

  return doc_length_array

def format_search_results(sorted_top_k, news_title_url):
  counter = 0

  if len((sorted_top_k)) == 0:
    print('no results found')
    return 

  else:
    for top in sorted_top_k:
      title = news_title_url[top[0]][0]
      url = news_title_url[top[0]][1]
      print('{}. document number: {} with cosine similarity:{}'.format(counter+1, top[0], top[1]))
      print('url : {}'.format(url))
      print('title : {}'.format(title))

      counter += 1

      if counter < len(sorted_top_k): 
        print('------------------------------------------------------')

def search_query(query, positional_index, doc_num, news_title_url):
 
  doc_scores = np.zeros(doc_num)
  doc_length_array = calculate_doc_length(positional_index, doc_num)
 
  # how many of the query words does each doc have
  doc_query_frequency = np.zeros(doc_num)

  query_terms = {}
  processed_query = query_preprocessing(query)
  
  # calculating tf for query terms in the given query 
  for query_term in processed_query:
    if query_term in query_terms:
      term_query_wieght += 1
    else:
      query_terms[query_term] = 1

  # index elimination
 
  for term in query_terms:
    if positional_index.get(term) is not None:
      term_positings_list = positional_index[term][1]

      for doc in term_positings_list:
        doc_query_frequency[doc] += 1
      else:
        continue
  
  for term in query_terms:
    # checking if the query term exists in dictionary
    if positional_index.get(term) is not None:
      
      term_positings_list = positional_index[term][1]
     
      term_query_weight = (1 + math.log10(query_terms[term])) 

      for doc in term_positings_list:
        # only considering doc with at least n-1 query terms
        if doc_query_frequency[doc] >= (len(query_terms) - 1):

          tf = 1 + math.log10(positional_index[term][1][doc])
          idf = math.log10(doc_num / positional_index[term][0])
          term_doc_weigth = (tf * idf)
          doc_scores[int(doc)] += term_query_weight * term_doc_weigth

        else:
          continue

      else:
        continue
 
  normalized_score = {}
  for doc in range(doc_num):
      if doc_length_array[doc] != 0.0:
        normalized_score[doc] = doc_scores[doc] / doc_length_array[doc]

  candidates = {}
  for doc, doc_score in sorted(normalized_score.items(), key=lambda item: item[1], reverse=True):
    candidates[doc] = doc_score

  top_k_docs = list(candidates.items())[:10]
  
  final_top_k = []
  for doc in top_k_docs:
    if doc[1] != 0.0:
      final_top_k.append(doc)

  return final_top_k

sorted_top_k = search_query('تیم ملی فوتبال', champions_list, len(news_title_url), news_title_url)
format_search_results(sorted_top_k, news_title_url)