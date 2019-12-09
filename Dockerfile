FROM tensorflow/tensorflow:2.0.0-gpu-py3



# RUN apt-get update && apt-get install python3-pip -y

COPY . /app
RUN python -m pip install -r /app/requirements.txt

RUN python -m nltk.downloader wordnet
