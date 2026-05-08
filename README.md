# GraphAnalysis

We propose a methodology to automatically find possibly polarized Echo Chambers on the massive Twitter/X network by using a community detection clustering algorithm on a heterogeneous graph of **user-hashtag**

The proposed framework is based on a strategy composed of five steps, comprising both network and text analysis, which are the two main approaches for echo chamber detection. The five steps,  are the following: 1) Twitter/X crawling, 2) Graph building, 3) Community clustering, 4) polarization, sentiment and topic analysis, and 5) statistical and manual review.

In the first step, _Twitter/X crawling_, the entire Italian Twitter/X stream beginning from February 2022 and ending on November 2022 is downloaded using the methodology based on 400 Italian stopwords and the Twitter language filter. An average of 1.6 million tweets a day were filtered using this methodology.

In the second step, _graph building_, the Twittersphere is modeled with an heterogeneous graph with two different kinds of nodes: user and hashtag nodes. There are three types of arcs: user-user, when an user retweets another user or a user reply/mention another user; user-hashtag, when an user employs an hashtag in his tweet, user-hashtag where an user retweet a tweet that include one or more hashtag, and hashtag-hashtag, when two hashtag occurs in the same tweet.

In the third step, _Community clustering_, the graph was partitioned in communities using the Leiden algorithm. The resulting communities are made up of both the user nodes and the hashtags.

In the fourth step, _sentiment, polarisation and topic analysis,_ starting from the text, is under development.

### 

## Added the *Twitter Linguistic Analysis* notebook. 

All the analysis can be run on Cleaned_twitter.csv if is on the same folder. The first block in the notebook is to clean up Raw Tweet and process them into the cleaned format. If using openAI as representation model for BERTopic (clearer topic titles) a openAI API keys should be provided. Otherwise a local transformer can be used (just uncomment the lines). 

## Unit test
To run tests, execute this command:
```bash
python .\tests\test_GraphGenerationUser.py
```

## Container
Step 1: Build the images
```bash
podman build -t graph-analysis -f ./docker/Dockerfile .
```

Step: 2: Run the containers
```bash
podman run \
  -v ./resources:/app/resources:ro \
  -v ./output:/app/output \
  -v ./properties:/app/properties \
  graph-analysis
```
