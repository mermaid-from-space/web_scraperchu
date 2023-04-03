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

# This section of the code defines a maximum number of tokens, creates a function called split_into_many 
# that takes text and a maximum number of tokens as input and splits the 
# text into chunks of a maximum number of tokens.
# It then loops through the dataframe and either adds the text to the list of
# shortened texts or splits the text into chunks of a maximum number of tokens and 
# adds the chunks to the list of shortened texts.

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    
shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )


# This section of the code creates a new dataframe from the list of shortened texts, 
# applies the tokenizer to the text column of the dataframe
#  to get the number of tokens for each row, and creates a histogram of the number of tokens per row

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()



# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()


# loading the embeddings from the DataFrame and converting them to numpy arrays.
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()

# functions that use the embeddings to find the most similar context to a question and then answer 
# it based on that context. These functions leverage 
# OpenAI’s language models and the embeddings created in Step 10 to provide accurate and reliable answers
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    
    
    returns = []
    cur_len = 0
    
    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# unction to answer two different questions. The first question is a simple one, 
# while the second question requires more specific knowledge. 
# This example demonstrates the versatility of the Q&A bot and its ability to answer a wide range of questions.


print(answer_question(df, question="What day is it?", debug=False))

print(answer_question(df, question="What is our newest embeddings model?"))        