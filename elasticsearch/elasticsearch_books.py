#!/usr/bin/env python
# coding: utf-8

__author__ = "Haifeng Zhao"
__email__ =  "piscarias@gmail.com"

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from urllib.request import urlopen
import json
import csv
import time

# connect DB
client = Elasticsearch(
    "http://elasticsearch-elasticsearch-1:9200"
)
client.info()

# Create the index
client.indices.delete(index="book_index", ignore_unavailable=True)
# Define the mapping
mappings = {
    "properties": {
        "book_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": "true",
            "similarity": "cosine"
        }
    }
}

client.indices.create(index='book_index', mappings=mappings)

# index the data
model = SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')

t_start = time.time_ns() / 1000000
operations = []
count = 0
with open('/home/zhf/rag/milvus/book/toy_books_5000.csv', encoding='Latin-1') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        operations.append({"index": {"_index": "book_index"}})
        combined_book_fields = row["Book-Title"] + " " + row["Book-Author"] + " " + row["Year-Of-Publication"] + " " + row["Publisher"]
        row["book_vector"] = model.encode(combined_book_fields).tolist()
        operations.append(row)
        count += 1
        if count % 1000 == 0:
            print("appended {cnt} books to index".format(cnt=count))
client.bulk(index="book_index", operations=operations, refresh=True)
t_end = time.time_ns() / 1000000
print("took {ms}ms to build the index with {num} records".format(ms=(t_end-t_start), num=client.count()['count']))

# the following code is to perform a vector search
response = client.search(
    index="book_index",
    knn={
      "field": "book_vector",
      "query_vector": model.encode("The Future Just Happened"),
      "k": 20,
      "num_candidates": 10000,
    },
    size=50
)

def pretty_response(response):
    if len(response['hits']['hits']) == 0:
        print('Your search returned no results.')
    else:
        for hit in response['hits']['hits']:
            id = hit['_id']
            publication_date = hit['_source']['Year-Of-Publication']
            score = hit['_score']
            title = hit['_source']['Book-Title']
            publisher = hit["_source"]["Publisher"]
            ISBN = hit["_source"]["ISBN"]
            authors = hit["_source"]["Book-Author"]
            pretty_output = (f"\nTitle: {title} ** Authors: {authors} ** Publisher: {publisher}  ** Year: {publication_date}")
            print(pretty_output)

pretty_response(response)

