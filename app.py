import requests
import requestsimport urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd 
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np 
from openai.embeddings_utils import distances_from_embeddings,
cosine_similarity

# it says regex pattern to match a url...idk what this means and i put it in parenthesis idk if i should have done that.
HTTP_URL_PATTERN = "r'^http[s]*://.+'"

#define root domain to crawl
domain = "openai.com"
full_url = "https://openai.com/"

#class to parse HTML and get hyperlinks
class HyperlinkParser(HTMLParser):
    def__init__(self):
        super().__init__()
        #create list to store hyperlinks
        self.hyperlinks
