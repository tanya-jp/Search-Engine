# -*- coding: utf-8 -*-

!pip install hazm
!pip3 install parsivar
from google.colab import drive
from parsivar import Tokenizer, Normalizer, FindStems
from hazm import utils
import json
import re
from copy import deepcopy
import functools
import numpy as np
import matplotlib.pyplot as plt
import math

drive.mount('/content/drive')

f = open("/content/drive/MyDrive/IR_data_news_12k.json")
news_dataset = json.load(f)
#check
for i in range(5):
  print(news_dataset[str(i)]['title'])

print(len(news_dataset))
f.close()

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
        # print(stemmed_token)
        stemmed_token_list.append(stemmed_token[0])
      elif mode == 3:
        stemmed_token_list.append((token, int(docID)))
    return stemmed_token_list


def preprocessing(news_dataset):
   tokenizer = Tokenizer()
   normalizer = Normalizer(statistical_space_correction=True)
   
   term_docID = []
   news_title_url = {}
   token_count_zipf = {}
   token_count_zipf_no_stopword = {}
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
     # calculating token_count_zipf with stopwords
     for i in range(len(tokenized_content)):
       count = token_count_zipf.get(tokenized_content[i],0)
       token_count_zipf[tokenized_content[i]] = count + 1
     # removing stopwords
     stopwords = utils.stopwords_list()

     for token in deepcopy(tokenized_content):
        if token in stopwords:
            tokenized_content.remove(token)
     # calculating token_count_zipf_no_stopword after removing stopwords
     for i in range(len(tokenized_content)):
       count = token_count_zipf_no_stopword.get(tokenized_content[i],0)
       token_count_zipf_no_stopword[tokenized_content[i]] = count + 1

     # stemming 
     term_docID.extend(stemming(tokenized_content, docID, 1))
    
   return term_docID, news_title_url, token_count_zipf, token_count_zipf_no_stopword

term_docID, news_title_url, token_count_zipf, token_count_zipf_no_stopword = preprocessing(news_dataset)

# Dictionary to keep the length of each doc 
# for calculating doc ranking in boolean query based on density
doc_len = {}

for token in term_docID:
  if token[1] not in doc_len:
    doc_len[token[1]] = 1
  else:
    doc_len[token[1]] = doc_len[token[1]] + 1

def positional_indexing(term_docID):
  positional_index = {}
  cnt = 0

  pre_docID = 0
  position = 1

  for item in term_docID:
    # print(item)
    token = item[0]
    docID = item[1]
    if pre_docID != docID:
      position = 1

    position_dic = {} 
    if token not in positional_index:     
      position_dic[docID] = [1, position]
      positional_index[token] = [1, position_dic]

      # print(positional_index)
    else:
      value = positional_index[token]
      position_dic = value[1]

      if docID not in position_dic:  
        rep = value[0]
        rep += 1
        value[0] = rep
        position_dic[docID] = [0, position]
      else:
        position_dic[docID].append(position)
      
      position_dic[docID][0] += 1
      

      positional_index[token] = value
    
    position += 1
    pre_docID = docID
    cnt += 1
    if cnt % 200000 == 0:
      print(cnt)

  # save the distionary to check it
  try:
    positional_index_file = open('positional_index.txt', 'wt')
    positional_index_file.write(str(positional_index))
    positional_index_file.close()
  
  except:
    print("Unable to write to file")
  
  return positional_index

positional_index = positional_indexing(term_docID)

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

# testing query_preprocessing function
query_content = 'نتایج جام ملت ها چه. می شود می روند می خواهم میخواهیم بخریم بخرم با هم!؟'
preprocessed_query = query_preprocessing(query_content)
print(preprocessed_query)

def simple_query_processing(query_content, positional_index):
  preprocessed_query = query_preprocessing(query_content)
  # key -> docID, value -> number of query words this doc contains
  res = {}
  for token in preprocessed_query:
    if token in positional_index:
      docs = list(positional_index[token][1].keys())
      for doc in docs:
        if doc not in res:
          res[doc] = 1
        else:
          res[doc] += 1

  sortedDict = sorted(res, key=res.get, reverse=True)

  final_res = []
  # key -> docID, value -> density of query words in the doc
  doc_density = {}
  for i in range (len(preprocessed_query), -1, -1):
    for doc in sortedDict:
      if res[doc] == i:
        sum = 0
        for token in preprocessed_query:
          if token in positional_index and doc in positional_index[token][1]:
            sum += positional_index[token][1][doc][0]
        doc_density[doc] = sum/doc_len[doc]
      else:
        final_res.extend(sorted(doc_density, key=doc_density.get, reverse=True))
        doc_density.clear()
  
  return res, final_res


# test
query_content = 'کارگران ایران خودرو'
raw_res, result = simple_query_processing(query_content, positional_index)
cnt = 0
if len(result) == 0:
  print('no results found')
else:
  for output in result:
      if cnt < 5:
        print(news_title_url[output])
        cnt += 1
      else:
        break

def not_query_processing(query_content, positional_index):
  query = query_content.split("!")
  not_terms = []
  for i in range(1, len(query)):
    if i < len(query):
      notquery = "!" + query[i].split()[0]
      not_terms.append(query[i].split()[0].replace("!", ""))
      query_content = query_content.replace(notquery, '')

  preprocessed_query = query_preprocessing(query_content)
  res, final_res = simple_query_processing(query_content, positional_index)

  # remove docs contain not terms
  for not_term in not_terms:
    if not_term in positional_index:
      for doc in deepcopy(final_res):
        if doc in positional_index[not_term][1]:
          final_res.remove(doc)

  return final_res


# test
query_content = ' کارگران !ایران !دولت'
# positional_index = positional_indexing(term_docID)
result = not_query_processing(query_content, positional_index)
cnt = 0
if len(result) == 0:
  print('no results found')
else:
  for output in result:
      if cnt < 5:
        print(news_title_url[output])
        cnt += 1
      else:
        break

def phrase_query_processing(query_content, positional_index):
  preprocessed_query = query_preprocessing(query_content)

  raw_res, result = simple_query_processing(query_content, positional_index)
    
  doc_density = {}
  for doc in raw_res:
    if raw_res[doc] == len(preprocessed_query):
      sum = 0
      for i in range(len(preprocessed_query)-1):
        term = preprocessed_query[i]
        next_term = preprocessed_query[i+1]

        if term in positional_index and next_term in positional_index:
          term_pos = positional_index[term][1][doc]
          next_term_pos = positional_index[next_term][1][doc]

          n = 1
          flag = 0
          for t in range(1, len(term_pos)):
            del_flag = 0
            while term_pos[t] + 1 != next_term_pos[n] and n + 1 < len(next_term_pos):
              n += 1

            if term_pos[t] + 1 == next_term_pos[n]:
              sum += 1
              flag = 1
            else:
              flag = 0

          if flag == 1:
            if i>0 and doc not in doc_density:
              continue
            else:
              doc_density[doc] = sum/doc_len[doc]
              flag = 0
          elif flag == 0 and doc in doc_density:
            del doc_density[doc]


  # #ranking
  sortedDict = sorted(doc_density, key=doc_density.get, reverse=True)
  return sortedDict



# test
query_content = 'صنعت نفت '
# positional_index = positional_indexing(term_docID)
result = phrase_query_processing(query_content, positional_index)
cnt = 0
if len(result) == 0:
  print('no results found')
else:
  for output in result:
      if cnt < 5:
        print(news_title_url[output])
        cnt += 1
      else:
        break

def complex_query_processing(query_content, positional_index):
  query = query_content.split("!")
  not_terms = []
  for i in range(1, len(query)):
    if i < len(query):
      notquery = "!" + query[i].split()[0]
      not_terms.append(query[i].split()[0].replace("!", ""))
      query_content = query_content.replace(notquery, '')
  
  quotation_term = query_content.split('"')[1::2]
  for term in quotation_term:
      query_content = query_content.replace(term, '')
  
  preprocessed_query = query_preprocessing(query_content)
  main_query_content = []
  if len(preprocessed_query)>0 and len(quotation_term)>0:
    preprocessed_query.extend(quotation_term)
    main_query_content = preprocessed_query
  elif len(preprocessed_query)>0:
    main_query_content = preprocessed_query
  elif len(quotation_term)>0:
    main_query_content = quotation_term
  # print(main_query_content)

  res = {}
  for term in quotation_term:
    temp = phrase_query_processing(term, positional_index)
  
    for doc in temp:
      if doc not in res:
          res[doc] = 1
      else:
          res[doc] += 1

  raw_res, result = simple_query_processing(query_content, positional_index)

  for doc in raw_res:
    if doc in res:
      res[doc] += raw_res[doc]
    else:
      res[doc] = raw_res[doc]

  #ranking
  sortedDict = sorted(res, key=res.get, reverse=True)

  result=[]
  result.extend(sortedDict)

  for not_term in not_terms:
    if not_term in positional_index:
      for doc in result:
        if doc in positional_index[not_term][1]:
          sortedDict.remove(doc)

  final_res = []
  doc_density = {}
  for i in range (len(main_query_content), -1, -1):
    flag = 0
    for doc in sortedDict:
      if res[doc] == i:
        sum = 0
        for token in main_query_content:
          # print(doc, token, quotation_term)
          if token in positional_index and doc in positional_index[token][1]:
            sum += positional_index[token][1][doc][0]
          elif token in quotation_term:
            # print("**")
            sum += 1
        doc_density[doc] = sum/doc_len[doc]
      elif res[doc] == i+1 and flag == 0:
        final_res.extend(sorted(doc_density, key=doc_density.get, reverse=True))
        doc_density = {}
        flag = 1
        
  return final_res



# test
query_content = '"صنعت نفت" !اروپا'
result = complex_query_processing(query_content, positional_index)
cnt = 0
if len(result) == 0:
  print('no results found')
else:
  for output in result:
      if cnt < 5:
        print(news_title_url[output])
        cnt += 1
      else:
        break


# Zipf law - before removing stopwords

# sort tokens based on count 
sorted_tokens_by_count = {}
for token, token_count in sorted(token_count_zipf.items(), reverse=True, key=lambda item: item[1]):
  sorted_tokens_by_count[token] = token_count

x_stop = [math.log10(y) for y in list(range(1,len(sorted_tokens_by_count)+1))]
y_stop = [math.log10(y) for y in list(sorted_tokens_by_count.values())]
y_ideal = [math.log10(list(sorted_tokens_by_count.values())[0]) - x for x in x_stop]

plt.plot(x_stop, y_stop, color='royalblue')
plt.plot(x_stop, y_ideal, color='pink', label='ideal')
plt.xlabel("log10 rank")
plt.ylabel("log10 cf")
plt.title("Before removing Stopwords")
leg = plt.legend(loc='best')
plt.show()

# Zipf law -  after removing stopwords

# sort tokens based on count 
sorted_tokens_by_count_ns = {}
for token, token_count in sorted(token_count_zipf_no_stopword.items(), reverse=True, key=lambda item: item[1]):
  sorted_tokens_by_count_ns[token] = token_count

y_ns = [math.log10(y) for y in list(sorted_tokens_by_count_ns.values())]
x_ns = [math.log10(y) for y in list(range(1, len(sorted_tokens_by_count_ns) + 1))]
y_ideal_ns = [math.log10(list(sorted_tokens_by_count_ns.values())[0]) - x for x in x_ns]

plt.plot(x_ns, y_ns, color='royalblue')
plt.plot(x_ns, y_ideal_ns, color='pink', label='ideal')
plt.xlabel("log10 rank")
plt.ylabel("log10 cf")
plt.title("After removing Stopwords")
leg = plt.legend(loc='best')
plt.show()

def preprocessing_ns(news_dataset):
   tokenizer = Tokenizer()
   normalizer = Normalizer(statistical_space_correction=True)
   
   term_docID_ns = []
   counter = 0
   for docID in news_dataset:
     if int(docID) % 1000 == 0:
       print(docID)
     content = news_dataset[docID]['content']
     new_content = re.sub(r'[^\w\s]', '', content)
     # normalize new content
     normalized_content = normalizer.normalize(new_content)
     # getting the tokens(non-positional)
     tokenized_content = tokenizer.tokenize_words(normalized_content)
     # calculating token_count_zipf with stopwords
     for i in range(len(tokenized_content)):
       count = token_count_zipf.get(tokenized_content[i],0)
       token_count_zipf[tokenized_content[i]] = count + 1
     # removing stopwords
     stopwords = utils.stopwords_list()

     for token in deepcopy(tokenized_content):
        if token in stopwords:
            tokenized_content.remove(token)
    
     # stemming 
     term_docID_ns.extend(stemming(tokenized_content, docID, 3))
    
   return term_docID_ns
term_docID_ns = preprocessing_ns(news_dataset)

def positional_indexing_ns(term_docID_ns):
  positional_index_ns = {}
  cnt = 0

  pre_docID = 0
  position = 1

  for item in term_docID_ns:
    token = item[0]
    docID = item[1]
    if pre_docID != docID:
      position = 1

    position_dic = {} 
    if token not in positional_index_ns:     
      position_dic[docID] = [1, position]
      positional_index_ns[token] = [1, position_dic]

    else:
      value = positional_index_ns[token]
      position_dic = value[1]

      if docID not in position_dic:  
        rep = value[0]
        rep += 1
        value[0] = rep
        position_dic[docID] = [0, position]
      else:
        position_dic[docID].append(position)
      
      position_dic[docID][0] += 1
      

      positional_index_ns[token] = value
    
    position += 1
    pre_docID = docID
    cnt += 1
    if cnt % 200000 == 0:
      print(cnt)

  # save the distionary to check it
  try:
    positional_index_file_ns = open('positional_index_ns.txt', 'wt')
    positional_index_file_ns.write(str(positional_index_ns))
    positional_index_file_ns.close()
  
  except:
    print("Unable to write to file")
  
  return positional_index_ns

positional_index_ns = positional_indexing_ns(term_docID_ns)

# Heaps law
def heap_dict_len(positional_index, heaps_dic):
    for instance in heaps_dic:
        for word in positional_index:
            for doc in positional_index[word][1]:
                if int(doc)<= instance:
                    heaps_dic[instance][0]+=1
                    break
    return heaps_dic

def heap_tokens_len(tokens, heaps_dic):
    for token in tokens:
        for doc_num in heaps_dic:
            if int(token[1])<= doc_num:
                heaps_dic[doc_num][1]+=1
    return heaps_dic

stemmed_dict = {500: [0, 0], 1000: [0, 0], 1500: [0, 0], 2000: [0, 0]}
stemmed = heap_dict_len(positional_index,stemmed_dict)
stemmed = heap_tokens_len(term_docID,stemmed)

not_stemmed_dict =  {500:[0,0], 1000:[0,0], 1500:[0,0], 2000:[0,0]}
not_stemmed = heap_dict_len(positional_index_ns, not_stemmed_dict)
not_stemmed = heap_tokens_len(term_docID_ns, not_stemmed)

# not stemmed
x_ns = np.array([math.log10(t[1]) for t in list(not_stemmed.values())])
y_ns = [math.log10(t[0]) for t in list(not_stemmed.values())]
mn, bn = np.polyfit(x_ns, y_ns, 1)
# stemmed
x = np.array([math.log10(t[1]) for t in list(stemmed.values())])
y = [math.log10(t[0]) for t in list(stemmed.values())]
m, b = np.polyfit(x, y, 1)

plt.plot(x_ns, mn*x_ns+bn, color='purple', label='no stemming')
plt.xlabel("log10 T")
plt.ylabel("log10 M")

plt.plot(x, m*x + b, color='royalblue', label='stemming')
plt.xlabel("log10 T")
plt.ylabel("log10 M")

plt.title("Vocabulary size computed based on heaps law")
leg = plt.legend(loc='best')
plt.show()

tokens_count = len(term_docID)
dict_count = len(positional_index)
tokens_ns_count = len(term_docID_ns)
dict_ns_count = len(positional_index_ns)

print("Heaps law vocabulary size prediction: "+ str(round(((10**b)*(tokens_count**m)),0)))
print("Vocabulary size with stemming: "+ str(dict_count))
print("k = "+ str(10**b)+ "  b = " + str(round(m,3)))
print('-----------------------------------------------------------')
print("Heaps law vocabulary size prediction: "+ str(round(((10**bn)*(tokens_ns_count**mn)),0)))
print("Vocabulary size without stemming: " + str(dict_ns_count))
print("k = " + str(10 ** bn) + "  b = " + str(round(mn, 3)))