# syntax=docker/dockerfile:1

FROM python:3.9

EXPOSE 5000

WORKDIR /app

COPY requirements.txt /app/requirements.txt
 
RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader wordnet
 		
COPY . /app

ENTRYPOINT python app.py