#!/usr/bin/python
# -*- coding: utf-8 -*-

from locust import HttpUser, task
from pymilvus import connections, FieldSchema, CollectionSchema, \
    DataType, Collection, utility
from sentence_transformers import SentenceTransformer


class QuickstartUser(HttpUser):

    host = 'http://milvus-standalone:19530'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = \
            SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base'
                                )
        self.search_params = {
            'metric_type': 'COSINE',
            'offset': 0,
            'ignore_growing': False,
            'params': {'nprobe': 1},
            }

    @task
    def queryAuthor(self):
        js = {
            'collectionName': 'book_search',
            'vector': self.model.encode('JOHN SAUL').tolist(),
            'limit': 20,
            'outputFields': ['book_title', 'book_author',
                             'year_of_publication', 'publisher'],
            'offset': 0,
            'param': self.search_params,
            }
        response = self.client.post('/v1/vector/search', json=js,
                                    headers={'Authorization': 'Bearer root:Milvus'
                                    })

    @task
    def queryTitle(self):
        js = {
            'collectionName': 'book_search',
            'vector': self.model.encode('The Mummies of Urumch'
                    ).tolist(),
            'limit': 20,
            'outputFields': ['book_title', 'book_author',
                             'year_of_publication', 'publisher'],
            'offset': 0,
            'param': self.search_params,
            }
        response = self.client.post('/v1/vector/search', json=js,
                                    headers={'Authorization': 'Bearer root:Milvus'
                                    })

    @task
    def queryPublisher(self):
        js = {
            'collectionName': 'book_search',
            'vector': self.model.encode('Oxford University Press'
                    ).tolist(),
            'limit': 20,
            'outputFields': ['book_title', 'book_author',
                             'year_of_publication', 'publisher'],
            'offset': 0,
            'param': self.search_params,
            }
        response = self.client.post('/v1/vector/search', json=js,
                                    headers={'Authorization': 'Bearer root:Milvus'
                                    })

