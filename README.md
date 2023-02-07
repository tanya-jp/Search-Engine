# Search-Engine
Persian Search Engine, Project of Information Retrieval Course

This project is an implementation of an information retrieval system using a large dataset of a Persian news agency documents.

## [Phase 1](https://github.com/tanya-jp/Search-Engine/blob/master/IR_phase1.py)

### Document Preprocessing

- Normalization

- Extracting Tokens
- Removing Stopwords
- Stemming
- Checking Heaps' Law and Zipf's Law Before and After Stemming

Result of Zipf's Law:
<p align="left">
  <img src="https://user-images.githubusercontent.com/72709191/217290254-635d6696-1f3f-4bad-9e99-686d3e6cdf84.png" width=35% height=35%>
</p>
<p align="left">
  <img src="https://user-images.githubusercontent.com/72709191/217290290-2d600027-bfff-403f-8cf3-29f90448da69.png" width=35% height=35%>
</p>
Result of Heaps' Law:
<p align="left">
  <img src="https://user-images.githubusercontent.com/72709191/217289460-efa3e028-2d58-4dac-b63c-bfbe7f8a3942.JPG" width=35% height=35%>
</p>

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
