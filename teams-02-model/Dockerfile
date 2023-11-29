FROM python:3.9

ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt \
	&& rm -Rf /var/cache/apt \
	&& rm -Rf /requirements.txt \
	&& rm -Rf /root/.cache/pip

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

ENV PYTHONPATH=/app

ADD * /app/

ENTRYPOINT [ "python3", "/app/baseline.py", "-i", "$inputDataset/inputs.jsonl", "-o", "$outputDir", "-m", "/app/model.pkl" ]
