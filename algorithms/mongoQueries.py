# This file stores all the functions related to MongoDB queries for the graph generation process.
"""
Extracting every tweets that is whenever a retweet, a reply or contains hashtags.
"""
def extract_tweets_for_user_graph():
    w = {} # we consider all tweets.
    s = {'_id': 0,
            'id': 1,
            'in_reply_to_status_id': 1,
            'in_reply_to_user_id': 1,
            'possibly_sensitive': 1,
            'created_at': 1,
            'source': 1,
            'userMentionEntities': 1,
            'hashtagEntities': 1,

            'user.id': 1,
            'user.friends_count': 1,
            'user.followers_count': 1,
            'user.listed_count': 1,
            'user.favourites_count': 1,
            'user.verified': 1,
            'user.created_at': 1,
            'user.screen_name': 1,
            'user.url': 1,
            'user.statuses_count': 1,
            'user.geo_enabled': 1,

            'retweeted_status.id': 1,
            'retweeted_status.retweet_count': 1,
            'retweeted_status.favourited_count': 1,

            'retweeted_status.user.id': 1,
            'retweeted_status.user.friends_count': 1,
            'retweeted_status.user.listed_count': 1,
            'retweeted_status.user.favourites_count': 1,
            'retweeted_status.user.statuses_count': 1,
            'retweeted_status.user.followers_count': 1,
        }

    return w, s

def extract_tweets_if_contains_hashtags_or_is_retweet_or_reply():
    w = {'$or': [{'hashtagEntities': {'$exists': True}}, {'retweeted_status': {'$exists': True}},
                    {'in_reply_to_status_id': {'$exists': True}}]}
    s = {'_id': 0,
            'id': 1,
            'in_reply_to_status_id': 1,
            'in_reply_to_user_id': 1,
            'retweeted_status.created_at': 1,
            'retweeted_status.id': 1,
            'retweeted_status.user.id': 1,
            'retweeted_status.user.screen_name': 1,
            'user.id': 1,
            'user.screen_name': 1,
            'hashtagEntities': 1,
            'created_at': 1,
            'userMentionEntities': 1}

    return w, s