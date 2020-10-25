FROM python:3.6

RUN apt-get update -y

WORKDIR /app

COPY . /app

RUN python -m pip install -r requirements.txt