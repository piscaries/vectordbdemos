#!/usr/bin/python
# -*- coding: utf-8 -*-
from locust import HttpUser, between, task
import time
from sentence_transformers import SentenceTransformer


class QuickstartUser(HttpUser):

    host = 'http://elasticsearch-elasticsearch-1:9200'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = \
            SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base'
                                )

    @task
    def queryTitle(self):
        eb = self.model.encode('The Mummies of Urumchi').tolist()
        js = {
           "knn":{
              "field":"book_vector",
              "query_vector": eb,
              "k":20,
              "num_candidates":10000
           },
           "fields":[
              "Book-Title",
	      "Book-Author",
	      "Publisher",
	      "Year-Of-Publication"
           ],
           "size":50
        }

        response = self.client.post('/book_index/_search', json=js)

    @task
    def queryAuthor(self):
        eb = self.model.encode('John Saul').tolist()
        js = {
           "knn":{
              "field":"book_vector",
              "query_vector": eb,
              "k":20,
              "num_candidates":10000
           },
           "fields":[
              "Book-Title",
	      "Book-Author",
	      "Publisher",
	      "Year-Of-Publication"
           ],
           "size":50
        }

        response = self.client.post('/book_index/_search', json=js)

    @task
    def queryPublisher(self):
        eb = self.model.encode('Oxford University Press').tolist()
        js = {
           "knn":{
              "field":"book_vector",
              "query_vector": eb,
              "k":20,
              "num_candidates":10000
           },
           "fields":[
              "Book-Title",
	      "Book-Author",
	      "Publisher",
	      "Year-Of-Publication"
           ],
           "size":50
        }

        response = self.client.post('/book_index/_search', json=js)



