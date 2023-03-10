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
    def __init__(self):
        super().__init__()
        #create list to store hyperlinks
        self.hyperlinks = []

    #override HTMLParser handle_starttag method to get hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        #if the tag is a anchor tag and has a href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyerlinks.append(attrs["href"])

    #this section of the code defines a function called get_hyperlinks that takes a URL as input, tries to open the URL
    # and read the HTML, and then parses the HTML to get hyperlinks. If the respone is not HTML it returns an empty list!

    #function get hyperlinks from a URL
    def get_hyperlinks(url):
        try:
            with urllib.request.urlopen(url) as response:
                if not response.info().get('Content-Type').startswith("text/html"):
                    return []

                #decode the html
                html = response.read().decode('utf-8')
        except Exception as e:
            print(e)
            return []

        #create the HTML parser and then parse the html to get hyperlinks
        parser = HyperlinkParser()
        parser.feed(html)

        return parser.hyperlinks
    