FROM python:3.7

WORKDIR /app

RUN pip install flask flask_restful numpy nltk scikit-learn
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

ADD ./models ./models
ADD app.py app.py

EXPOSE 5000

CMD ["python", "app.py"]