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

# fuction to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_domain_hyperlinks(url)):
        clean_link = none

        #if the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            #Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link 
        
        #if the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

#return the list of hyperlinks that are within the same domain
return list(set(clean_links))


def crawl(url):
    #parse the URL and get the domain
    local_domain = urlparse(url).netloc

    #create a queue to store the URLS to crawl
    queue = deque([url])

    #create a set to store the urls that have already been seen (no duplicates)
    seen = set([url])

    # create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exits("text/"+local_domain+"/"):
        os.mkdir("text/"+ local_domain + "/")

    # create directory to store the cvs files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # while the que is not empty, continue crawling
    while queue:

        #get the next URL from the queue
        url = queue.pop()
        print(url) #for debugging and to see the progress

        #save text from the url to a <url>.txt file
        with open("text/"+local_domain+"/"+url[8:].replace("/","_") + ".txt", "w", encoding="UTF-8") as f:

            # get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # get the text but remove the tags
            text = soup.get_text()

            # if the crawler gets to a page that requires Javascript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + "due to JavaScript being required")

            # otherwise, write the text to the file in the text directory
            f.write(text)

        # get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in see:
                queue.append(link)
                seen.add(link)

crawl(full_url)


#this section of code defines a function called remove_newlines that takes a pandas series object
# as input, replaces newlines with spaces, and returns the modified series.
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

   # Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

# This section of the code loads a tokenizer and applies it to the text column of the dataframe 
# to get the number 
# of tokens for each row. It then creates a histogram of the number of tokens per row.

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()