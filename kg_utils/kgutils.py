# -*- coding: utf-8 -*- 
import collections
import json
import logging
import re
import sys
import random 
from kg_utils.queries import preprocess




REPLACEMENTS = [
    [' < ', ' math_lt '],
    [' > ', ' math_gt '],
    ["<", ">", ""],#
    ['dbo:', 'http://dbpedia.org/ontology/', 'dbo_'],
    ['dbp:', 'http://dbpedia.org/property/', 'dbp_'],
    ['dbc:', 'http://dbpedia.org/resource/Category:', 'dbc_'],
    ['dbr:', 'res:', 'http://dbpedia.org/resource/', 'dbr_'],
    ['dct:', 'dct_'],
    ['geo:', 'geo_'],
    ['georss:', 'georss_'],
    ['rdf:', 'rdf_'],
    ['rdfs:', 'rdfs_'],
    ['foaf:', 'foaf_'],
    ['owl:', 'owl_'],
    ['yago:', 'yago_'],
    ['skos:', 'skos_'],
    [' ( ', '  par_open  '],
    [' ) ', '  par_close  '],
    ['(', ' attr_open '],
    [') ', ')', ' attr_close '],
    ['{', ' brack_open '],
    ['}', ' brack_close '],
    [' . '," ; ", ' sep_dot '],#
    ['. ',"; ", ' sep_dot '],#
    ['?', 'var_'],
    ['*', 'wildcard'],
    [' <= ', ' math_leq '],
    [' >= ', ' math_geq '],
    #
    [":", "_"]
]


STANDARDS = {
        'dbo_almaMater': ['dbp_almaMater'],
        'dbo_award': ['dbp_awards'],
        'dbo_birthPlace': ['dbp_birthPlace', 'dbp_placeOfBirth'],
        'dbo_deathPlace': ['dbp_deathPlace', 'dbp_placeOfDeath'],
        'dbo_child': ['dbp_children'],
        'dbo_college': ['dbp_college'],
        'dbo_hometown': ['dbp_hometown'],
        'dbo_nationality': ['dbo_stateOfOrigin'],
        'dbo_relative': ['dbp_relatives'],
        'dbo_restingPlace': ['dbp_restingPlaces', 'dbp_placeOfBurial', 'dbo_placeOfBurial', 'dbp_restingplace'],
        'dbo_spouse': ['dbp_spouse'],
        'dbo_partner': ['dbp_partner']
}


def encode( sparql ):
    encoded_sparql = do_replacements(sparql)
    shorter_encoded_sparql = shorten_query(encoded_sparql)
    normalized = normalize_predicates(shorter_encoded_sparql)
    normalized = str(" ").join(str(normalized).split())
    return normalized


def decode ( encoded_sparql ):
    short_sparql = reverse_replacements(encoded_sparql)
    sparql = reverse_shorten_query(short_sparql)
    return sparql


def normalize_predicates( sparql ):
    for standard in STANDARDS:
        for alternative in STANDARDS[standard]:
            sparql = sparql.replace(alternative, standard)

    return sparql


def do_replacements( sparql ):
    for r in REPLACEMENTS:
        encoding = r[-1]
        for original in r[:-1]:
            sparql = sparql.replace(original, encoding)
    return sparql


def reverse_replacements( sparql ):
    for r in REPLACEMENTS:
        original = r[0]
        encoding = r[-1]
        sparql = sparql.replace(encoding, original)
        stripped_encoding = str.strip(encoding)
        sparql = sparql.replace(stripped_encoding, original)
    return sparql


def shorten_query( sparql ):
    sparql = re.sub(r'order by desc\s+....?_open\s+([\S]+)\s+....?_close', '_obd_ \\1', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'order by asc\s+....?_open\s+([\S]+)\s+....?_close', '_oba_ \\1', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'order by\s+([\S]+)', '_oba_ \\1', sparql, flags=re.IGNORECASE)
    return sparql


def reverse_shorten_query( sparql ):
    sparql = re.sub(r'_oba_ ([\S]+)', 'order by asc (\\1)', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'_obd_ ([\S]+)', 'order by desc (\\1)', sparql, flags=re.IGNORECASE)
    return sparql


def fix_URI(query):
	query = re.sub(r"dbr:([^\s]+)" , r"<http://dbpedia.org/resource/\1>" , query)
	if query[-2:]=="}>":
		query = query[:-2]+">}"
	return query


def select(query):
    if "SELECT " in query:
        sparql =  str(list(str(query).split("SELECT "))[-1])
        sparql = "SELECT "+ sparql
        return sparql
    elif "ASK " in query:
        sparql =  str(list(str(query).split("ASK "))[-1])
        sparql = "ASK "+ sparql
        return sparql
    else:
        return query
    
    
def get_en(question_dict):
    for quest in question_dict:
        if quest["language"] == "en":
            return str(quest["string"]).lower().strip() ;
    
    
def interprete(enocoded_sentence):
    decoded_sparql = decode(enocoded_sentence)
    query = fix_URI(decoded_sparql)
    return str(query) 


