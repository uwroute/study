#encoding=utf-8
import sys,os

def emit(key, value):
    print key"\t"value

def mapper(k, v):
    cols = v.split(' ')
    item = cols[1]
    score = cols[2]
    emit(item, score)

def reducer(k, vs):
