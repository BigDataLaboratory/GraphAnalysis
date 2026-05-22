import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.GraphGenerationUser import GraphGenerationUser
from Utils.Utils import Utils

class TestGraphGenerationUser(unittest.TestCase):
    
    @patch('algorithms.MongoConnection.MongoClient')
    def setUp(self, mock_mongo_client):
        # We mock MongoClient to avoid real connection when instantiating GraphGenerationUser
        self.mock_client = MagicMock()
        mock_mongo_client.return_value = self.mock_client
        
        self.output_path = "./test_output"
        self.gg_user = GraphGenerationUser(
            uri="fake_uri",
            database_name="fake_db",
            collection="fake_collection",
            output_file_path=self.output_path
        )
    
    def test_process_user_tweets_empty(self):
        features, edges_rt, edges_reply, mention_raw = self.gg_user.process_user_tweets(12345, [])
        self.assertIsNone(features)
        self.assertEqual(edges_rt, [])
        self.assertEqual(edges_reply, [])
        self.assertEqual(mention_raw, [])
        
    def test_process_user_tweets_extraction(self):
        # Create fake tweets for a user
        now = datetime.now()
        fake_tweets = [
            {
                "id": 1,
                "user": {
                    "id": 100, 
                    "screen_name": "user1", 
                    "followers_count": 50,
                    "friends_count": 10,
                    "favourites_count": 5,
                    "verified": False,
                    "created_at": datetime(2020, 1, 1)
                },
                "created_at": datetime(2020, 1, 2),
                "retweeted_status": None,
                "in_reply_to_status_id": -1,
                "hashtagEntities": "AI|Graph",
                "userMentionEntities": "user2"
            },
            {
                "id": 2,
                "user": {
                    "id": 100, 
                    "screen_name": "user1", 
                    "followers_count": 55, # most recent tweet metrics
                    "friends_count": 10,
                    "favourites_count": 6,
                    "verified": False,
                    "created_at": datetime(2020, 1, 1)
                },
                "created_at": datetime(2020, 1, 3), # more recent
                "retweeted_status": {
                    "id": 10,
                    "user": {"id": 200},
                    "created_at": datetime(2020, 1, 1)
                },
                "in_reply_to_status_id": -1,
                "hashtagEntities": "Graph",
                "userMentionEntities": ""
            },
            {
                "id": 3,
                "user": {
                    "id": 100, 
                    "screen_name": "user1", 
                    "followers_count": 50,
                    "friends_count": 10,
                    "favourites_count": 5,
                    "verified": False,
                    "created_at": datetime(2020, 1, 1)
                },
                "created_at": datetime(2020, 1, 1),
                "retweeted_status": None,
                "in_reply_to_status_id": 999,
                "in_reply_to_user_id": 300,
                "hashtagEntities": "",
                "userMentionEntities": "user3|user4"
            }
        ]
        
        user_id = 100
        features, edges_rt, edges_reply, mention_raw = self.gg_user.process_user_tweets(user_id, fake_tweets)
        
        # Test Features
        self.assertIsNotNone(features)
        self.assertEqual(features['user_id'], 100)
        self.assertEqual(features['total'], 1.39)
        self.assertEqual(features['retweets'], 0.69)
        self.assertEqual(features['replies'], 0.69)
        self.assertEqual(features['original'], 0.69)
        self.assertEqual(features['followers'], 4.03) # Check that the most recent features were captured
        self.assertEqual(features['n_unique_hashtags'], 2)  # ai, graph
        self.assertEqual(features['n_unique_mentions'], 3) # user2, user3, user4
        
        # Test Edges Retweet
        self.assertEqual(len(edges_rt), 1)
        self.assertEqual(edges_rt[0][1], 200)
        self.assertEqual(edges_rt[0][2], 1)
        
        # Test Edges Reply
        self.assertEqual(len(edges_reply), 1)
        self.assertEqual(edges_reply[0][1], 300)
        self.assertEqual(edges_reply[0][2], 1)
        
        # Test Mention (user2, user3, user4 but 3 and 4 are not retweeted so counted)
        # Note: In process_user_tweets, it ignores mentions if it's a retweet.
        # Tweet 1 is original -> user2
        # Tweet 2 is retweet -> no mentions
        # Tweet 3 is reply -> user3, user4
        self.assertEqual(len(mention_raw), 3)

if __name__ == '__main__':
    unittest.main()