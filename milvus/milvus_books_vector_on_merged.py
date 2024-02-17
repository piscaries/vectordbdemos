#!/usr/bin/env python
# coding: utf-8

__author__ = "Haifeng Zhao"
__email__ =  "piscarias@gmail.com"

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import ops, pipe
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import csv

connections.connect(host='milvus-standalone', port='19530')

utility.drop_collection("book_search")

# create collection
def create_milvus_collection(collection_name, dim=768):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=20),
            FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=500), 
            FieldSchema(name="book_author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="year_of_publication", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="publisher", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="combined_fields", dtype=DataType.VARCHAR, max_length=710),
            FieldSchema(name="combined_fields_vector", dtype=DataType.FLOAT_VECTOR,dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='Book-Info')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type': "COSINE",
        'index_type': "IVF_SQ8",
        'params': {"nlist": 1}
    }
    collection.create_index(field_name='combined_fields_vector', index_params=index_params)
    return collection

collection = create_milvus_collection('book_search')

# Index data
insert_pipe = (pipe.input('id', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'combined_fields') 
               .map('combined_fields', 'combined_fields_vector', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
               .map('combined_fields_vector', 'combined_fields_vector', lambda x: x / np.linalg.norm(x, axis=0))
               .map(('id', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'combined_fields', 'combined_fields_vector'), 'res', ops.ann_insert.milvus_client(host='milvus-standalone', port='19530', collection_name='book_search'))
               .output('res') )

t_start = time.time_ns() / 1000000
with open('/home/zhf/rag/milvus/book/toy_books_5000.csv', encoding='Latin-1') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    count = 0
    for row in reader:
        row = row[0:5]
        row.append(row[1] + " " + row[2] + " " + row[3] + " " + row[4])
        insert_pipe(*row)
        count += 1
    print(count)
t_end = time.time_ns() / 1000000
print("took {ms}ms to build the index for {num} records".format(ms=(t_end-t_start), num=collection.num_entities))


# the following code is for search purpose
collection.load()
model = SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
search_params = {
    "metric_type": "COSINE", 
    "offset": 0, 
    "ignore_growing": False, 
    "params": {"nprobe": 1}
}

results = collection.search(
    data=[model.encode("JOHN SAUL")], 
    anns_field="combined_fields_vector", 
    # the sum of `offset` in `param` and `limit` 
    # should be less than 16384.
    param=search_params,
    limit=20,
    expr=None,
    # set the names of the fields you want to 
    # retrieve from the search result.
    output_fields=['book_title', 'book_author', 'year_of_publication', 'publisher'],
    consistency_level="Strong"
)

def print_result(results):
    for hit in results[0]:
        print("Title: {title} ** Author: {author} ** Year: {year} ** Publisher: {publisher}\n".format(title=hit.entity.get('book_title'), \
                                                                                                      author=hit.entity.get('book_author'), \
                                                                                                      year=hit.entity.get('year_of_publication'), \
                                                                                                      publisher=hit.entity.get('publisher')))
print_result(results)
