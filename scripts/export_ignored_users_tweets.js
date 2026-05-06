// Usage: mongosh --quiet <db_name> export_ignored_users_tweets.js > ignored_users_tweets.json

const fs = require('fs');

// Path to the file containing ignored user IDs
const ignoredUserIdsPath = './ignored_user_ids.txt';

// Read and parse the IDs from the text file
const textData = fs.readFileSync(ignoredUserIdsPath, 'utf8');
const targetUserIds = textData
    .split('\n')
    .map(id => id.trim())
    .filter(id => id.length > 0)
    .map(id => Number(id)); // Convert to JS Number

const projection = {
    '_id': 0,
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
};

// Filter the tweets to only include the target user IDs, then group by user.id
const cursor = db.QCPS_2.find({ "user.id": { $in: targetUserIds } }, projection);

while (cursor.hasNext()) {
    const doc = cursor.next();

    // Print the tweet in Extended JSON format to preserve BSON types like Longs ($numberLong)
    print(EJSON.stringify(doc));
}
