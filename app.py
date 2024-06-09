from flask import Flask, request, render_template
import requests
import os
from bs4 import BeautifulSoup
import torch
from transformers import pipeline
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set environment variables to control TensorFlow behavior and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Function to fetch and parse HTML content
def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    logging.debug("Fetched HTML content successfully")
    return BeautifulSoup(response.content, 'html.parser')

# Function to extract main text content from HTML
def extract_text(soup):
    paragraphs = soup.find_all('p', class_='pw-post-body-paragraph')
    text = ' '.join([p.get_text() for p in paragraphs])
    logging.debug("Extracted text successfully")
    return text

# Function to preprocess text
def preprocess_text(text):
    logging.debug("Preprocessed text successfully")
    return ' '.join(text.split())

# Function to split the text into smaller chunks if it's too long
def split_text(text, max_length=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logging.debug("Split text into chunks successfully")
    return chunks

# Function to summarize text using Hugging Face's Transformers
def summarize_text(text, max_length=1024, batch_size=4):
    text_chunks = split_text(text, max_length=max_length)
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summaries = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_summaries = summarizer(batch, max_length=150, min_length=30, do_sample=False)
        summaries.extend([summary['summary_text'] for summary in batch_summaries])
    final_summary = " ".join(summaries)
    logging.debug("Summarized text successfully")
    return final_summary

# Main function to summarize web content
def summarize_web_content(url):
    soup = fetch_html(url)
    text = extract_text(soup)
    preprocessed_text = preprocess_text(text)
    summary = summarize_text(preprocessed_text)
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        logging.debug(f"Received URL: {url}")
        summary = summarize_web_content(url)
        return render_template('index.html', summary=summary)
    return render_template('index.html', summary=None)

if __name__ == '__main__':
    app.run(debug=True)
