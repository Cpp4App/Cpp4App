import csv
import re
import spacy
from bs4 import BeautifulSoup

def get_alifornia(text):
    specialArea = ""
    california = 0
    with open(text, encoding='utf-8') as file_obj:
        for line in file_obj:
            specialArea += line
    if "alifornia" in specialArea:
        california = 1
    return specialArea,california


import sys
maxInt = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


def get_text(path):
    htmlfile = open(path, 'r', encoding='utf-8')
    htmlhandle = htmlfile.read()

    soup = BeautifulSoup(htmlhandle, 'html.parser')

    stri = str(soup)
    return stri




