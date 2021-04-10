# CORD-19_articles_ELasticsearch_engine
Created a preprocessing pipeline in Python for CORD-19 articles from Kaggle ( https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) to perform a search using Elasticsearch.
Compared two Information Retrieval systems in order to determine whether "Keyword Selection" and "Text Lemmatisation" steps improve search performance in terms of relevancy of documents.

The first Information Retrieval system is a full system from the assignment 1 that comprises such steps to perform a search:
•	Data Loading
•	Text Normalisation
•	Text Lemmatisation
•	Data Indexing
•	Searching in ElasticSearch
The second Information Retrieval system is a system that, unlike the first system, neither selects keywords  from sentences nor lemmatises words. Thus, its pipeline includes the following steps:
•	 Data Loading
•	Text Normalisation
•	Data Indexing
•	Searching in ElasticSearch
The reason for composing the second system is to examine if an exclusion of “Keywords selection” and “Text Lemmatisation” steps from the whole sequence means a deterioration of search accuracy.
This hypothesis implies that a text of an original state is harder to be searched for due to the large amount of unnecesary words that occur in a majority of documents. Thus, the probability of obtaining irrelevant documents with the same words from a query is very high. 
In contrast to it, a processed text, that contains only lemmatised keywords, delivers all essential information while having a brief form that is more convenient for a search.

In table 1 all precision and recall values for each of the query and each of the system are represented.
Table 1 – All retrieved results for each query and each system
	System 1	System 2
	P@5	R@5	P@5	R@5
Q1	0.6	0.5	0.6	0.6
Q2	0.8	0.667	0.6	0.5
Q3	0.6	0.429	0.6	0.6

According to Table 1 above, it can be noticed that the search performance of the system 2,  from where steps “Keywords selection” and “Text Lemmatisation” were excluded,  is quite stable for each of the query and doesn’t exhibit any sharp fluctuations. 
In contrast to it, the behaviour of the first system is more unpredictable. For the second query, it is obvious that precision and recall values are higher than the System 2 ones. This fact indicates that the search performance of the first system, which includes “Keywords selection” and “Text Lemmatisation” steps, is better. 
On the other hand, according to the results for the other queries, its search accuracy is slightly worse. This indication can point out that particularly the “Keywords selection” step can remove some really important words. In this case, a search efficiency may be decreased simply due to a lack of words in texts. Therefore, for some cases,  this step may not work properly, since it depends on many words that were removed.
Although, considering the distance between each relevant document among the retrieved ones from the first system it is apparent that these documents are grouped more densely.  This interesting detail points out that despite having a lack of words in texts, the first system puts more purpose into its returned results.
All in all, considering the obtained results, it can be stated that a comparison of these two IR systems requires more deep research that includes building more queries or/and testing other configurations of the first system to determine their real efficiency on a larger set of data.
