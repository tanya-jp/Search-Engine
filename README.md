# Search-Engine
Persian Search Engine, Project of Information Retrieval Course

This project is an implementation of an information retrieval system using a large dataset of a Persian news agency documents.

## [Phase 1](https://github.com/tanya-jp/Search-Engine/blob/master/IR_phase1.py)

### Document Preprocessing

- Normalization

- Extracting Tokens
- Removing Stopwords
- Stemming
- Checking Heaps' law and Zipf's law before and after stemming
### Positional index construction

### Query processing

- Normal Queries
- Phrase Queries
- Boolean Queries (NOT)
- Combination of All Types of Queries

## [Phase 2](https://github.com/tanya-jp/Search-Engine/blob/master/IR_phase2.py)
The aim of this part is representing queries and documents as vectors to achieve more precise results. Documents are ranked according to their proximity to the query in the vector space. 
This includes the following steps:

1. tf–idf weighting
2. Ranking documents based on cosine similarity
3. Speeding cosine computation by pruning:
  - Index elimination
  - Champion list
