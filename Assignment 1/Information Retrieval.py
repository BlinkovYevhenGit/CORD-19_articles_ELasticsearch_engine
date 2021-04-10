#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
# os.system('pip install unidecode')
# os.system('pip install gensim')
# os.system('pip install flashtext')
# os.system('pip install pyspark')
# os.system('pip install tabulate')
# In[2]:


import re
import traceback
from pyspark.sql import SparkSession
from elasticsearch import Elasticsearch
import requests, json
from nltk.corpus import stopwords
import string
import unidecode
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from flashtext import KeywordProcessor
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import pprint
from IPython.display import display
from tabulate import tabulate
import datetime
# In[3]:
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import desc


s_top_words = set(stopwords.words('english'))


# ## Data loading

# In[4]:

def check_date_format(string_date):
    fixed_string=None
    try:
        fixed_string=str(datetime.datetime.strptime(string_date, '%Y-%m-%d').strftime('%Y-%m-%d'))
    except ValueError:
        print(string_date," - incorrect data format, should be YYYY-MM-DD")
        return None
    return fixed_string

def load_data(path = 'metadata.csv', n=1000):
    spark = SparkSession \
    .builder \
    .appName("ElasticSpark-1") \
    .config("spark.driver.extraClassPath", "/path/elasticsearch-hadoop-7.6.2/dist/elasticsearch-spark-20_2.11-7.6.2.jar") \
    .config("spark.es.port","9200") \
    .config("spark.driver.memory", "8G") \
    .config("spark.executor.memory", "12G") \
    .getOrCreate()

    metadata_df = spark.read.csv(path, multiLine=True, header=True)
    metadata_df.show(1)
    metadata_df = metadata_df.withColumn("index", monotonically_increasing_id())

    metadata_df=metadata_df.orderBy(desc("index")).drop("index").limit(n)
#     metadata_df = metadata_df.select("*").limit(1000)
    metadata_df.show(3)
    metadata_table = metadata_df.toPandas()
    print("Data loaded.")
    display(metadata_table)
    metadata_table["pr_title"]=metadata_table["title"]
    metadata_table["pr_abstract"]=metadata_table["abstract"]
    metadata_table["publish_time"]=metadata_table["publish_time"].apply(lambda string_date: check_date_format(string_date))
    return metadata_table


# In[5]:


def create_index(es_index="covid"):
    es = init_Elasticsearch_session()
    mapping = {
        'settings': {
            'number_of_shards': 1,
            'number_of_replicas': 1
        },
        'mappings': {
            "date_detection": "false",
            'properties': {
                'cord_uid': {
                    'index': 'false',
                    'type': 'text'
                },
                'sha': {
                    'index': 'true',
                    'type': 'text'
                },
                'source_x': {
                    'index': 'true',
                    'type': 'text'
                },
                'title': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'pr_title': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'doi': {
                    'index': 'true',
                    'type': 'text'
                },
                'pmcid': {
                    'index': 'true',
                    'type': 'text'
                },
                'license': {
                    'index': 'true',
                    'type': 'text'
                },
                'abstract': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'pr_abstract': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'publish_time': {
                    'index': 'true',
                    'type': 'date'
                },
                'authors': {
                    'index': 'true',
                    'type': 'text'
                },
                'journal': {
                    'index': 'true',
                    'type': 'text'
                },
                'who_covidence_id': {
                    'index': 'true',
                    'type': 'text'
                },
                'arxiv_id': {
                    'index': 'true',
                    'type': 'text'
                },
                'pdf_json_files': {
                    'index': 'true',
                    'type': 'text'
                },
                'pmc_json_files': {
                    'index': 'true',
                    'type': 'text'
                },
                'url': {
                    'index': 'true',
                    'type': 'text'
                },
                's2_id': {
                    'index': 'true',
                    'type': 'text'
                }
            }
        }
    }
    if es.indices.exists(es_index):
        es.indices.delete(es_index)

    es.indices.create(index=es_index, body=mapping)
    return es


def init_Elasticsearch_session():
    res = requests.get('http://localhost:9200')
    print(res.content)
    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    return es


# ## Sentence Splitting, Tokenization and Normalization
# In[6]:

class TextNormalizer:
    def __init__(self):
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def normalize_text(self, text):
        if text == None:
            return None
        try:
            normalized_sentences = []
            text = re.sub(' +', ' ', text)
            text = unidecode.unidecode(text)
            text = text.lower()
            sentences = sent_tokenize(text)
        except:
            print("ERROR:", text)
            traceback.print_exc()
            return None

        for sentence in sentences:
            # remove punctuation
            sentence = re.sub("[" + string.punctuation + "\d*]", " ", sentence)
            # strip leading/trailing whitespace
            sentence = sentence.strip()
            words = word_tokenize(sentence)
            new_sentence = ' '.join(words)  # we want to keep it as before to extract phrases
            words.append(new_sentence)
            normalized_sentences.append(words)
        return normalized_sentences


# In[7]:


def normalize_table(metadata_table):
    normaliser = TextNormalizer()

    table_to_process = metadata_table[["pr_title", "pr_abstract"]]
    table_to_process["pr_title"] = table_to_process["pr_title"].apply(lambda x: normaliser.normalize_text(x))
    table_to_process["pr_abstract"] = table_to_process["pr_abstract"].apply(lambda x: normaliser.normalize_text(x))

    for i in range(0, len(table_to_process)):
        metadata_table.loc[i, "pr_title"] = table_to_process.loc[i, "pr_title"]
        metadata_table.loc[i, "pr_abstract"] = table_to_process.loc[i, "pr_abstract"]
    return metadata_table


# ## Selecting key words

# In[8]:


def remove_stop_words(text):
    if text==None:
        return
    for index,sentence in enumerate(text):
        sentence = sentence[-1].split(" ") #performing tokenisation
        words = [word for word in sentence if word not in s_top_words and len(word)>2]
        sentence=" ".join(words)
        text[index]=sentence
    return text


# In[9]:


def get_words_corpus(table):
    words_corpus = []
    for i in range(0, len(table)):
        row = table.loc[i]
        title_sentences = row["pr_title"]
        abstract_sentences = row["pr_abstract"]

        if title_sentences != None:
            for i in range(0, len(title_sentences)):
                words_corpus.extend(title_sentences[i].split())

        if abstract_sentences != None:
            for i in range(0, len(abstract_sentences)):
                words_corpus.extend(abstract_sentences[i].split())
    return words_corpus


# In[10]:


def get_keywords_by_textrank(sentences):
    if sentences==None:
        return None
    keywords=dict()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank", last=True)
    doc = nlp(" ".join(sentences))

    # examine the top-ranked phrases in the document

    for p in doc._.phrases:
        if p.rank>=0.05:
            keywords[p.text]=p.rank
    #         print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    #         print(p.text)
    return keywords

# In[11]:


def extract_keywords(text,keyword_processor):
    sentences=[]
    if text==None:
        return None
    for i in range(0, len(text)):
        keywords_found = keyword_processor.extract_keywords(text[i])
        sentences.append(" ".join(keywords_found))
    return sentences

# In[12]:


def merge_two_keywords_methods(sentences, text_rank_key_word_processor, frequent_key_words_processor):
    if sentences==None:
        return None
    text_rank_version = extract_keywords(sentences,text_rank_key_word_processor)
    frequent_key_words_version = extract_keywords(sentences,frequent_key_words_processor)
    merged_version=list()
    for i in range(0, len(text_rank_version)):
        sentence_text_rank=text_rank_version[i].split()
        sentence_frequent_key_words=frequent_key_words_version[i].split()
        intersect = set(sentence_frequent_key_words) - set(sentence_text_rank)
        merged_words=sentence_text_rank + list(intersect)
        merged_sentence=" ".join(merged_words)
#         merged_words.append(merged_sentence)
        merged_version.append(merged_sentence)
    return merged_version


# In[13]:


def retain_best_tf_idf_keywords(sentences, index, tfIdf,tfIdfVectorizer):
    if sentences==None:
        return None
    tf_idf_keyword_processor = KeywordProcessor()
    df = pd.DataFrame(tfIdf[index].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF_IDF"])
    df = df.sort_values('TF_IDF', ascending=False)
    df = df[df.TF_IDF>0.09]
    tf_idf_dict=df.T.to_dict('list')
    for keyword in tf_idf_dict.keys():
        parts = " ".join(keyword.split("_"))
        tf_idf_keyword_processor.add_keyword(keyword,parts)
    sentences = extract_keywords(sentences,tf_idf_keyword_processor)
    new_sentences=[]
    for i, sentence in enumerate(sentences):
        if len(sentence)!=0:
            words=sentence.split()
            new_sentence=words
            new_sentence.append(sentence)
            new_sentences.append(new_sentence)
    return new_sentences


# In[14]:


def select_best_keywords(metadata_table):
    table_to_process = metadata_table[["pr_title", "pr_abstract"]]
    table_to_process["pr_title"] = table_to_process["pr_title"].apply(lambda x: remove_stop_words(x))
    table_to_process["pr_abstract"] = table_to_process["pr_abstract"].apply(lambda x: remove_stop_words(x))

    print("Text Data after removing of stop-words")
    display(table_to_process)

    words_corpus = get_words_corpus(table_to_process)
    print(len(words_corpus))

    dist = nltk.FreqDist(words_corpus)  # Creating a distribution of words' frequencies
    grams = dist.most_common(1000)  # Obtaining the most frequent words
    bigrams = nltk.collocations.BigramAssocMeasures()
    trigrams = nltk.collocations.TrigramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words_corpus)
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(words_corpus)

    print("Showing first", 2000, "top-freqent words in the corpus")
    grams = pd.DataFrame(grams)
    grams.index = range(1, len(grams) + 1)
    grams.columns = ["Word", "Frequency"]
    display(grams)

    bi_filter = 7
    print("Showing bigrams in the corpus found by Pointwise Mutual Information method")
    print("Applying frequency filter: a bigramm occurs more than", bi_filter, "times")
    bigramFinder.apply_freq_filter(bi_filter)
    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram', 'PMI']).sort_values(
        by='PMI', ascending=False)
    bigramPMITable["bigram"] = bigramPMITable["bigram"].apply(lambda x: ' '.join(x))
    display(bigramPMITable)

    tri_filter = 5
    print("Showing trigrams in the corpus found by Pointwise Mutual Information method")
    print("Applying frequency filter: a trigramm occurs more than", tri_filter, "times")
    trigramFinder.apply_freq_filter(tri_filter)
    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)),
                                   columns=['trigram', 'PMI']).sort_values(by='PMI', ascending=False)
    trigramPMITable["trigram"] = trigramPMITable["trigram"].apply(lambda x: ' '.join(x))
    display(trigramPMITable)

    gram_dict = grams.set_index('Word').T.to_dict('list')
    bigramPMIDict = bigramPMITable.set_index('bigram').T.to_dict('list')
    trigramPMIDict = trigramPMITable.set_index('trigram').T.to_dict('list')

    keyword_processor = KeywordProcessor()
    textrank_keyword_processor = KeywordProcessor()

    gram_dict.update(bigramPMIDict)
    bigramPMIDict.update(trigramPMIDict)

    #     print(gram_dict)
    print("Extracting keywords from texts using Pointwise Mutual Information method and TextRank")
    text_rank_key_words = dict()
    for i in range(0, len(table_to_process)):
        sentences = table_to_process.loc[i, "pr_abstract"]
        if sentences != None:
            keywords = get_keywords_by_textrank(sentences)
            if keywords != None:
                text_rank_key_words.update(keywords)
                print("Text", i, "- Done")
    for i in range(0, len(table_to_process)):
        sentences = table_to_process.loc[i, "pr_title"]
        if sentences != None:
            keywords = get_keywords_by_textrank(sentences)
            if keywords != None:
                text_rank_key_words.update(keywords)
                print("Text", i, "- Done")

    for keyword in gram_dict.keys():
        parts = keyword.split()
        parts = "_".join(parts)
        keyword_processor.add_keyword(keyword, parts)

    for keyword in text_rank_key_words.keys():
        parts = keyword.split()
        parts = "_".join(parts)
        textrank_keyword_processor.add_keyword(keyword, parts)

    print(len(keyword_processor.get_all_keywords()))
    print(len(textrank_keyword_processor.get_all_keywords()))
    print(len(text_rank_key_words))

    table_to_process["pr_abstract"] = table_to_process["pr_abstract"].apply(
        lambda x: merge_two_keywords_methods(x, textrank_keyword_processor, keyword_processor))
    table_to_process["pr_title"] = table_to_process["pr_title"].apply(
        lambda x: merge_two_keywords_methods(x, textrank_keyword_processor, keyword_processor))

    for i in range(0, len(table_to_process)):
        metadata_table.loc[i, "pr_title"] = table_to_process.loc[i, "pr_title"]
        metadata_table.loc[i, "pr_abstract"] = table_to_process.loc[i, "pr_abstract"]

    print("Comparison of Text Data after Keywords Extraction using Pointwise Mutual Information method and TextRank")
    display(metadata_table[["title", "pr_title", "abstract", "pr_abstract"]])

    print("Extracting keywords from texts using TF/IDF")
    dataset = []
    for i in range(0, len(table_to_process["pr_abstract"])):
        sentences = table_to_process.loc[i, "pr_abstract"]
        if sentences != None:
            sentences = " ".join(sentences)
            dataset.append(sentences)

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(dataset)

    index = 0
    for i in range(0, len(metadata_table)):
        if table_to_process.loc[i, "pr_abstract"] == None:
            continue
        metadata_table.loc[i, "pr_abstract"] = retain_best_tf_idf_keywords(table_to_process.loc[i, "pr_abstract"],
                                                                           index, tfIdf, tfIdfVectorizer)
        index += 1
    print("Extracting keywords from texts using TF/IDF")
    dataset = []
    for i in range(0, len(table_to_process["pr_title"])):
        sentences = table_to_process.loc[i, "pr_title"]
        if sentences != None:
            sentences = " ".join(sentences)
            dataset.append(sentences)

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(dataset)

    index = 0
    for i in range(0, len(metadata_table)):
        if table_to_process.loc[i, "pr_title"] == None:
            continue
        metadata_table.loc[i, "pr_title"] = retain_best_tf_idf_keywords(table_to_process.loc[i, "pr_title"], index,
                                                                        tfIdf, tfIdfVectorizer)
        index += 1
    return metadata_table


# ## Stemming or Morphological Analysis (Lemmatisation) 

# In[15]:


def lemmatise_text(sentences):
    if sentences==None:
        return None
    lemmatizer = WordNetLemmatizer()
    for i in range(0, len(sentences)):
        try:
            sentence=sentences[i][-1]
            if sentence == "":
                continue
            words=sentence.split()
            lemmatised_words = [lemmatizer.lemmatize(word) for word in words]
            new_sentence = ' '.join(lemmatised_words)
            lemmatised_words.append(new_sentence)
            sentences[i]=lemmatised_words
        except:
            print(sentences)
            print(sentences[i])
            traceback.print_exc()
            break
    return sentences


# ## Indexing

# In[16]:


def index_table(es,metadata_table,es_index="covid"):
    for i in range(0,len(metadata_table)):
        metadata_table.iloc[i].to_json(es_index+'.json')
        f = open(es_index+'.json')
        docket_content = f.read()
        row=json.loads(docket_content)
        try:
            es.index(index=es_index, id=i, body=row)
        except:
            traceback.print_exc()
            print("Error:", "row #"+str(i))


# ## Searching

# In[17]:


def search(es, es_index="covid",
    query={
          "query": {
            "match_phrase":{"publish_time":"2000-08-15"}
          }
        }):
    res = es.search(index=es_index, body=query)
    documents=[]
    for i in range(0, len(res['hits']['hits'])):
        doc=res['hits']['hits'][i]['_source']
        documents.append(doc)
    return documents


# ## Running of the program

# In[18]:


def run_program(path='metadata.csv', es_index="covid",
                query={
                    "query": {
                            "match_all": {}
                        }
                    },
                sequence=["Data loading","Data indexing","Searching in ElasticSearch","Text normalisation","Selecting keywords","Text lemmatisation","Data indexing","Searching in ElasticSearch"]):
    actions = {
        # Data loading
        "Data loading": { "function":lambda path: data_loading(path), "resources":["path"]}, #Returns metadata_table, Requires path
        # Indexing
        "Data indexing":{ "function":lambda es_index, metadata_table: data_indexing(es_index, metadata_table), "resources":["es_index", "metadata_table"]}, #Returns es
        # Sentence splitting, text tokenisation and normalisation
        "Text normalisation": {"function":lambda metadata_table: normalise_text(metadata_table),"resources":["metadata_table"]}, #Returns metadata_table
        # Selecting keywords
        "Selecting keywords": {"function":lambda metadata_table: select_keywords(metadata_table),"resources":["metadata_table"]}, #Returns metadata_table
        # Text lemmatisation
        "Text lemmatisation": {"function":lambda metadata_table: lemmatise_table(metadata_table),"resources":["metadata_table"]}, #Returns metadata_table
        # Searching in ElasticSearch
        "Searching in ElasticSearch": { "function":lambda es, es_index, query: return_searched_documents(es, es_index, query), "resources":["es","es_index", "query"]} #Returns documents
    }
    available_resources=dict()
    available_resources["path"]=path
    available_resources["es_index"]=es_index
    available_resources["query"]=query
    for action in sequence:
        func = actions[action]["function"]
        resources = actions[action]["resources"]
        missing_params=[]
        try:
            parameters = [available_resources[resource] if resource in available_resources else missing_params.append(resource) for resource in resources]
        except:
            print("There are not enough parameters to call the function", action, "!")
            print("Missing parameters: ",missing_params)
            return
        val1,context = func(*parameters)
        available_resources[context[0]] = val1
    return available_resources["metadata_table"], available_resources["documents"],available_resources["es"]

def abstract_method(resources, action):
    parameters = resources[action]
    context = action(*parameters)
    resources[context[1]]=context[0]

def return_searched_documents(es, es_index, query):
    print("Searching in ElasticSearch")
    documents = search(es, es_index=es_index, query=query)
    print("Retrieved documents:")
    # for document in documents:
    pprint.pprint(query)
    pprint.pprint(documents)
    print("Number of found documents - ",len(documents))
    print()
    context=["documents"]
    return documents,context


def lemmatise_table(metadata_table):
    print("Text lemmatisation")
    metadata_table["pr_abstract"] = metadata_table["pr_abstract"].apply(lambda x: lemmatise_text(x))
    metadata_table["pr_title"] = metadata_table["pr_title"].apply(lambda x: lemmatise_text(x))
    print("Comparison of Text Data after Applied Lemmatisation")
    print(metadata_table[["title", "pr_title", "abstract", "pr_abstract"]])
    context = ["metadata_table"]
    return metadata_table,context


def select_keywords(metadata_table):
    print("Selecting keywords")
    metadata_table = select_best_keywords(metadata_table)
    print("Comparison of Text Data after Selecting keywords step")
    print(metadata_table[["title", "pr_title", "abstract", "pr_abstract"]])
    context = ["metadata_table"]
    return metadata_table, context


def normalise_text(metadata_table):
    print("Sentence splitting, text tokenisation and normalisation")
    metadata_table = normalize_table(metadata_table)
    print("Comparison of Text Data after Sentence splitting, text tokenisation and normalisation step")
    print(metadata_table[["title", "pr_title", "abstract", "pr_abstract"]])
    context = ["metadata_table"]
    return metadata_table, context


def data_indexing(es_index, metadata_table):
    print("Creating index -", es_index)
    es = create_index(es_index)
    print("Indexing")
    index_table(es, metadata_table, es_index=es_index)
    print("Data indexed.")
    context = ["es"]
    return es, context


def data_loading(path):
    # Data loading
    print("Data loading")
    metadata_table = load_data(path)
    print("pr_title and pr_abstract columns have been added")
    print(metadata_table[["publish_time"]])
    context = ["metadata_table"]
    return metadata_table, context

# In[19]:

# System 1
metadata_table, documents, es = run_program(path='metadata.csv', es_index="covid3",sequence=["Data loading",
                                                                                             "Text normalisation",
                                                                                             "Selecting keywords",
                                                                                             "Text lemmatisation",
                                                                                             "Data indexing",
                                                                                             "Searching in ElasticSearch"])
print("Retrieved documents:")
pprint.pprint(documents)

# System 2
metadata_table, documents, es = run_program(path='metadata.csv', es_index="covid4",sequence=["Data loading",
                                                                                             "Text normalisation",
                                                                                             "Data indexing",
                                                                                             "Searching in ElasticSearch"])
print("Retrieved documents:")
pprint.pprint(documents)
#
# # In[20]:
# Enlist typical symptoms when having a Covid disease
es = init_Elasticsearch_session()
query1_covid1_docs,context = return_searched_documents(es,
                                      es_index="covid3",
                                      query={
                                          "size": "10",
                                          "query": {
                                              "multi_match":{
                                                  "query":"Main symptoms of a Covid disease",
                                                  "fuzziness" : "AUTO",
                                                  "fields" : [ "pr_title", "pr_abstract^5" ],
                                                  "type":"best_fields",
                                                  "analyzer": "standard" ,
                                                  "minimum_should_match":"50%"
                                              }
                                          },
                                            "_source": ["cord_uid","title","abstract","publish_time"],
                                      })
#Descrive an impact of diabetes, obesity, pulmonary diseases on acquiring a severe form of coronavirus disease
query2_covid1_docs,context = return_searched_documents(es,
                                      es_index="covid3",
                                      query={
                                          "size": "10",
                                          "query": {
                                              "multi_match": {
                                                  "query": "Influence of diabetes, obesity, pulmonary diseases on acquiring a severe form of Coronavirus disease",
                                                  "fuzziness":"AUTO",
                                                  "fields": ["pr_title", "pr_abstract^3" ],
                                                  "analyzer": "english",
                                              }
                                          },
                                          "_source": ["cord_uid","title","abstract","publish_time"],
                                      })
query3_covid1_docs,context = return_searched_documents(es,
                                        es_index="covid3",
                                                       query={
                                                           "size": "10",
                                                           "query": {
                                                               "bool": {
                                                                   "must": {
                                                                       "multi_match": {
                                                                           "query": "The effectiveness of wearing masks",
                                                                           "fields": ["pr_title", "pr_abstract^5"],
                                                                           "analyzer": "english",
                                                                            "minimum_should_match":"50%"
                                                                       }
                                                                   },
                                                                   "filter": {
                                                                       "range": {
                                                                           "publish_time": {
                                                                               "gte": "2020-06-01",
                                                                               "lte": "2021-03-01",
                                                                               "format": "year_month_day",
                                                                           }
                                                                       }
                                                                   }
                                                               }
                                                           },
                                                           "_source": ["cord_uid", "title", "abstract", "publish_time"],
                                                       })

query1_covid2_docs,context = return_searched_documents(es,
                                      es_index="covid4",
                                      query={
                                          "size": "10",
                                          "query": {
                                              "multi_match":{
                                                  "query":"Main symptoms of a Covid disease",
                                                  "fuzziness" : "AUTO",
                                                  "fields" : [ "pr_title", "pr_abstract^5"  ],
                                                  "type":"best_fields",
                                                  "analyzer": "standard",
                                                  "minimum_should_match":"50%"
                                              }
                                          },
                                            "_source": ["cord_uid","title","abstract","publish_time"],
                                      })
query2_covid2_docs,context = return_searched_documents(es,
                                      es_index="covid4",
                                      query={
                                          "size": "10",
                                          "query": {
                                              "multi_match": {
                                                  "query": "Influence of diabetes, obesity, pulmonary diseases on acquiring a severe form of Coronavirus disease",
                                                  "fuzziness":"AUTO",
                                                  "fields": ["pr_title", "pr_abstract^3" ],
                                                  "analyzer": "english",
                                              }
                                          },
                                          "_source": ["cord_uid","title","abstract","publish_time"],
                                      })
#Find the latest articles from June 2020 that describe the spread of coronavirus in the world
query3_covid2_docs,context = return_searched_documents(es,
                                      es_index="covid4",
                                                       query={
                                                           "size": "10",
                                                           "query": {
                                                               "bool": {
                                                                   "must": {
                                                                       "multi_match": {
                                                                            "query": "The effectiveness of wearing masks",
                                                                           "fields": ["pr_title", "pr_abstract^5"],
                                                                           "analyzer": "english",
                                                                            "minimum_should_match":"50%"

}
                                                                   },
                                                                   "filter": {
                                                                       "range": {
                                                                           "publish_time": {
                                                                               "gte": "2020-06-01",
                                                                               "lte": "2021-03-01",
                                                                               "format": "year_month_day",
                                                                           }
                                                                       }
                                                                   }
                                                               }
                                                           },
                                                           "_source": ["cord_uid", "title", "abstract", "publish_time"],
                                                       })


augmented_results={
    "query1":[],
    "query2":[],
    "query3":[]
}

systems_docs={"System_1":[query1_covid1_docs,query2_covid1_docs,query3_covid1_docs],"System_2":[query1_covid2_docs,query2_covid2_docs,query3_covid2_docs]}
def add_documents(systems_docs,augmented_results):
    for ind_query in range(0, len(systems_docs["System_1"])):
        for i in range(0, 10):
            inner_documents_dict = dict()
            for key in systems_docs["System_1"][0][0].keys():
                document_value1=systems_docs["System_1"][ind_query][i][key]
                document_value2=systems_docs["System_2"][ind_query][i][key]
                column_comparison={
                    key:{"System_1": document_value1,
                         "System_2": document_value2}
                }
                inner_documents_dict.update(column_comparison)
            augmented_results["query"+str(ind_query+1)].append(inner_documents_dict)


import webbrowser
from IPython.display import HTML
add_documents(systems_docs,augmented_results)
df = pd.json_normalize(augmented_results["query1"])
html_table = df.to_html(classes='table table-striped')
file1 = open("html_table1.html","w", encoding='utf-8')#write mode
file1.write(html_table)
file1.close()

url = 'html_table1.html'
webbrowser.open(url, new=2)  # open in new tab

df = pd.json_normalize(augmented_results["query2"])
html_table = df.to_html(classes='table table-striped')
# Write-Overwrites
file1 = open("html_table2.html","w", encoding='utf-8')#write mode
file1.write(html_table)
file1.close()

url = 'html_table2.html'
webbrowser.open(url, new=2)  # open in new tab

df = pd.json_normalize(augmented_results["query3"])
html_table = df.to_html(classes='table table-striped')
# Write-Overwrites
file1 = open("html_table3.html","w", encoding='utf-8')#write mode
file1.write(html_table)
file1.close()
url = 'html_table3.html'
webbrowser.open(url, new=2)  # open in new tab

documents = return_searched_documents(es, es_index="covid", query={
    "query": {
        "match_phrase": {"abstract": "biological diversity"}
    }
})

documents,context = return_searched_documents(es, es_index="covid3", query={
    "query": {
        "bool": {
          "filter": {
            "exists": {
              "field": "abstract"
            }
          },
          "must_not": {
            "term": {
              "abstract.keyword": "None"
            }
          }
        }
    },
    "size":"50",
    "_source": ["cord_uid","title","abstract"]
})


# System 1 System 2
# 0 1 - 1
# 1 0 - 1
# 2 0 - 1
# 3 1 - 1
# 4 1 - 0
# 5 1 - 1
# 6 1 - 0
# 7 1 - 1
# 8 0 - 0
# 9 0 - 0

# 0  1 - 1
# 1  0 - 1
# 2  0 - 0
# 3  1 - 1
# 4  1 - 0
# 5  1 - 1
# 6  1 - 0
# 7  1 - 1
# 8  0 - 0
# 9  0 - 0

# System 1 System 2
# 0 1 - 1
# 1 1 - 0
# 2 0 - 1
# 3 0 - 1
# 4 1 - 0
# 5 1 - 1
# 6 0 - 0
# 7 0 - 1
# 8 1 - 0
# 9 0 - 1

# System 1 System 2
# 0 1 - 0
# 1 0 - 1
# 2 0 - 1
# 3 1 - 0
# 4 1 - 1
# 5 1 - 1
# 6 1 - 1
# 7 0 - 0
# 8 1 - 0
# 9 1 - 0




def calc_precision_at_K(relevancy_results, K):
    precision_list=[]
    current_relevant_docs_amount=0
    precision_at_K=0
    for index in range(0, K):
        if relevancy_results[index]==1:
            current_relevant_docs_amount+=1
        precision_at_K=current_relevant_docs_amount/(index+1)
        precision_list.append(precision_at_K)
    last_item = precision_list[-1]
    return precision_list, last_item


def calc_recall_at_K(relevancy_results, K):
    recall_list=[]
    all_relevant_docs_amount=relevancy_results.count(1)
    current_relevant_docs_amount=0
    recall_at_K=0
    for index in range(0, K):
        if relevancy_results[index]==1:
            current_relevant_docs_amount+=1
        recall_at_K=current_relevant_docs_amount/(all_relevant_docs_amount)
        recall_list.append(recall_at_K)
    last_item=recall_list[-1]
    return recall_list, last_item

relevancy_results_query1_System1=[1,0,0,1,1,1,1,1,0,0] # Results for the Information Retrieval system without changes
relevancy_results_query1_System2=[1,1,0,1,0,1,0,1,0,0] #Results for the Information Retrieval system without keywords selection, lemmatisation

relevancy_results_query2_System1=[1,1,0,1,1,1,0,0,1,0] # Results for the Information Retrieval system without changes
relevancy_results_query2_System2=[1,0,1,1,0,1,0,1,0,1] #Results for the Information Retrieval system without keywords selection, lemmatisation

relevancy_results_query3_System1=[1,0,0,1,1,1,1,0,1,1] # Results for the Information Retrieval system without changes
relevancy_results_query3_System2=[0,1,1,0,1,1,1,0,0,0] #Results for the Information Retrieval system without keywords selection, lemmatisation
K=5
print("Results for the Information Retrieval system without changes")
precision_list, precision_at_K=calc_precision_at_K(relevancy_results_query1_System1, K)
recall_list,recall_at_K=calc_recall_at_K(relevancy_results_query1_System1, K)
print("Precision at K =",K," -",precision_at_K)
print("Recall at K =",K," -",recall_at_K)
pprint.pprint(precision_list)
pprint.pprint(recall_list)

print()
print("Results for the Information Retrieval system without keywords selection, lemmatisation")
precision_list, precision_at_K=calc_precision_at_K(relevancy_results_query1_System2, K)
recall_list,recall_at_K=calc_recall_at_K(relevancy_results_query1_System2, K)
print("Precision at K =",K," -",precision_at_K)
print("Recall at K =",K," -",recall_at_K)
pprint.pprint(precision_list)
pprint.pprint(recall_list)

