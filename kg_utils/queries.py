# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:09:29 2019

@author: Stuart
"""
import traceback
import sys
import importlib
import json
import requests
import numpy as np 
import xmltodict
from collections import OrderedDict
from SPARQLWrapper import SPARQLWrapper, JSON, XML

"""
 e.g.
 query1 = "ASK WHERE { <http://dbpedia.org/resource/Taiko> a <http://dbpedia.org/class/yago/WikicatJapaneseMusicalInstruments> }" 
 query2 = "SELECT DISTINCT ?n WHERE { <http://dbpedia.org/resource/Chiemsee> <http://dbpedia.org/ontology/maximumDepth> ?n }"
"""

def process(query):
    query = str(query).strip()
    try:
        response = getSPARQL(query)
    except:            
        response = getVirtuoso(query)
    #print("\n ", response)
    entity = preprocess(response)
    #!
    print()
    return entity ;


def preprocess(response):
    try:
        if "boolean" in response.keys():
            value = response["boolean"]
        else:
            result = response["results"]["bindings"] 
            value = [ list(item.values())[0]["value"] for item in result]
            if 0==len(value):
                return "empty"
        return value
    except :
       return "error";

       
def getVirtuoso(query):
    data={
            'query':query,
            'default-graph-uri':'http://dbpedia.org'
            }
    response = requests.post("https://dbpedia.org/sparql", data=data)
    try:
        content = response.json()
    except:
        content = response.text;
    try:
        content = xmltodict.parse(content)
        content = json.loads(json.dumps(content, indent=4, default=str))
    except:
        content = content
    return content;
 
    
def getSPARQL(query):    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    try:
        sparql.setReturnFormat(JSON)
        response = sparql.query().convert()
    except:
        sparql.setReturnFormat(XML)
        response = sparql.query().convert()
    return response;


#def lookup_embedding(entity):
#    if "http://" in entity:
#        entity = "<"+str(entity).strip()+">"
#        
#        try:
#            embed = model[entity]
#            return np.array(embed)
#        except:
#            #embed = np.zeros([200,]) 
#            return "!!!"
#    else:
#        return str(entity).strip()



