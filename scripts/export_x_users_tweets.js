// Usage: mongosh --quiet <db_name> export_100_users.js > sample_100_users.json

const MAX_USERS = 2000;

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

// Sort by user.id to group the tweets of each user
const cursor = db.QCPS_2.find({}, projection).sort({ "user.id": 1 });

let currentUserId = null;
let usersCount = 0;

while (cursor.hasNext()) {
    const doc = cursor.next();
    const userId = doc.user ? doc.user.id : null;

    if (!userId) continue;

    // If user.id changes
    if (userId !== currentUserId) {
        if (currentUserId !== null) {
            usersCount++;
        }
        currentUserId = userId;
    }

    // If we have exceeded the user quota, we stop
    if (usersCount >= MAX_USERS) {
        break;
    }

    // We print the tweet in Extended JSON format to preserve BSON Longs ($numberLong)
    print(EJSON.stringify(doc));
}