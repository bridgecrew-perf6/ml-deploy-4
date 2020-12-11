from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle
import re
import os
import numpy as np
from nltk.corpus import stopwords

app = Flask(__name__)
api = Api(app)

class Dataset(object):
    def __init__(self, timeline_list=None):
        self.timeline = [] # a list of unprocessed tweets (str)
        self.avg_emoticon_counts = None # additional features
        self.avg_hashtag_counts = None
        self.avg_exclamation_counts = None
        self.avg_reply_counts = None
        self.avg_repetition_counts = None
        self.avg_tweet_len = None
        self.processed_timeline = None
        
        if not timeline_list is None:
            for tweet_dict in timeline_list:
                self.timeline.append(tweet_dict["text"])
           
    def process(self):
        """
        make ready text and features for Models to use
        """
        self.processed_timeline = " ".join(list(map(self.process_tweet, self.timeline)))
        
        return [self.processed_timeline], [self.avg_emoticon_counts, self.avg_hashtag_counts, self.avg_exclamation_counts, self.avg_reply_counts, self.avg_repetition_counts, self.avg_tweet_len]
    
    def process_tweet(self, text, language='english'):
        """
        text processing for a single tweet
        """
        # for handling URLs and REPLY tags
        URL_REGEX = re.compile(r'(https?|ftp)://[^\s]*')
        REPLY_REGEX = re.compile(r'@username')
        URL_TAG = 'URL'
        REPLY_TAG = 'REP'
        
        text = URL_REGEX.sub(URL_TAG, text) # set URL tags
        text = REPLY_REGEX.sub(REPLY_TAG, text) # set REPLY tags
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text) # trim repeated chars
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        words = letters_only.lower().split()   # lower case                        
        stops = set(stopwords.words(language)) # remove stopwords                 
        meaningful_words = [w for w in words if not w in stops]  
        return(" ".join(meaningful_words))

    # avg number of emoticons
    def count_avg_emoticons(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(re.findall(r'(?::|;|:\'|=)(?:-)?(?:\)|\(|D|P|d|p|3)|<3|</3|xd|xD|XD', tweet))
            return count/len(self.timeline)

    # avg number of character repetitions
    def count_avg_repetitions(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(re.findall(r'(\w)\1{3,}', tweet))
            return count/len(self.timeline)

    # avg number of replies
    def count_avg_replies(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(re.findall(r'@username', tweet))
            return count/len(self.timeline)

    # avg number of hastags
    def count_avg_hashtags(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(re.findall(r'#(\w+)', tweet))
            return count/len(self.timeline)

    # avg number of exclamation marks
    def count_avg_exclamations(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(re.findall(r'!+', tweet))
            return count/len(self.timeline)
        
    # avg tweet length
    def count_avg_tweet_len(self):
        if not self.timeline is None:
            count = 0
            for tweet in self.timeline:
                count += len(tweet)
            return count/len(self.timeline)
        
    def generate_features(self):
        """
        Given timeline, generate features:
        (1) emoticons
        (2) character repetitions
        (3) hastags
        (4) exclamation mark
        (5) replies
        (6) length of tweet
        """
        self.avg_emoticon_counts = self.count_avg_emoticons()
        self.avg_hashtag_counts = self.count_avg_hashtags()
        self.avg_exclamation_counts = self.count_avg_exclamations()
        self.avg_reply_counts = self.count_avg_replies()
        self.avg_repetition_counts = self.count_avg_repetitions()
        self.avg_tweet_len = self.count_avg_tweet_len()

        
class Predictor(object):
    def __init__(self, timeline=None, features=None, personalities=['extroverted', 'stable', 'agreeable', 'conscientious', 'open']):
        self.personalities = personalities
        self.timeline = timeline
        self.features = features
        self.models = {}
        self.vectorizers = {}
     
        for personality in self.personalities:
            # load trait vectorizers
            if os.path.isfile(f'./models/{personality}_vectorizer.pkl'):
                self.vectorizers[personality] = pickle.load(open(f'./models/{personality}_vectorizer.pkl','rb'))
            # load trait models
            if os.path.isfile(f'./models/{personality}_model.pkl'):
                self.models[personality] = pickle.load(open(f'./models/{personality}_model.pkl','rb'))
                
    def load_input(self, trait=None):
        """
        make ready input vector for corresponding personality model
        """
        text_vec = self.vectorizers[trait].transform(self.timeline).toarray()
        input_vec = np.column_stack([text_vec, np.array([self.features])])
        return input_vec
    
    def estimate_traits(self):
        """
        make prediction for each personality trait
        """
        output = {}
        for personality in self.personalities:
            input_vec = self.load_input(personality)
            output[personality] = round(self.models[personality].predict(input_vec)[0], 2)
        return output
    
class Respond(Resource):
    @staticmethod
    def post():
        # BODY: {"posts": [ {"text": tweet}, {"text": tweet}, ...] }
        post_data = request.get_json()['posts']
        data = Dataset(post_data)
        data.generate_features()
        processed_timeline, features = data.process()
        
        predictor = Predictor(processed_timeline, features)
        output = predictor.estimate_traits()
        
        return jsonify(output)
        
api.add_resource(Respond, '/predictpersonality')

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0')#, debug=True)