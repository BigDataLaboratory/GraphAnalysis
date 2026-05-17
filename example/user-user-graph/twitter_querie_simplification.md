## extract 2000 root tweet
db.QCPS_2.find({
    "retweeted_status": { "$exists": false },
    "in_reply_to_status_id": { "$in": [-1, null] }},
    {_id: 0, id: 1}
).limit(2000)



## Analysing the extraction script
1. Fetch root tweet:
   - We only select 2000 for our local test
   - In the real database we will extract all of them
   - So no index needed here
   - However we can filter fields we only need to build our graph
2. Fetch all recursive children
   - we are looking for tweets containing parent tweet id (field: reply, retweet, mention)
   - Indexes needed: in_reply_to_status_id, retweeted_status.id, screen_name
3. Fetch all the authors tweet
   - we are looking for tweets written by a user
   - Indexes needed: user.id

We also need to filter which fields to include
### Indexes needed?
### Select only needed features?
#### Original json
{
    "_id": {
        "$oid": "62165d624330c45866817a5f"
    },
    "in_reply_to_status_id": 1496379176995246083,
    "possibly_sensitive": false,
    "userMentionEntities": "JmsFortuna",
    "created_at": {
        "$date": "2022-02-23T16:13:53Z"
    },
    "truncated": false,
    "source": "<a href=\"https://mobile.twitter.com\" rel=\"nofollow\">Twitter Web App</a>",
    "retweeted_status": {
        "in_reply_to_status_id": -1,
        "possibly_sensitive": false,
        "created_at": {
            "$date": "2022-02-23T06:59:31Z"
        },
        "truncated": false,
        "source": "<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>",
        "retweet_count": 226,
        "favourited_count": 325,
        "in_reply_to_screen_name": null,
        "in_reply_to_user_id": -1,
        "id": 1496379176995246083,
        "text": "Direi che non servono ulteriori commenti. https://t.co/1GBhSCzer2",
        "user": {
            "utc_offset": -1,
            "friends_count": 405,
            "listed_count": 1,
            "favourites_count": 878,
            "verified": false,
            "description": "Opinione. Un'idea che possedete; la convinzione è, invece, un'idea che possiede voi.\n(John Garland Pollard)",
            "created_at": {
                "$date": "2012-07-04T20:02:25Z"
            },
            "time_zone": null,
            "url": null,
            "screen_name": "JmsFortuna",
            "statuses_count": 2035,
            "followers_count": 329,
            "name": "James Fortuna",
            "location": null,
            "id": 626823042,
            "geo_enabled": true,
            "lang": null
        },
        "favorited": false
    },
    "retweet_count": 0,
    "favourited_count": 0,
    "in_reply_to_screen_name": null,
    "userMentionEntitiesArray": [
        "JmsFortuna"
    ],
    "in_reply_to_user_id": 626823042,
    "id": 1496518687930699780,
    "text": "RT @JmsFortuna: Direi che non servono ulteriori commenti. https://t.co/1GBhSCzer2 #MilanUdinese #SempreMilan",
    "hashtagEntitiesArray": [
        "MilanUdinese",
        "SempreMilan"
    ],
    "hashtagEntities": "MilanUdinese|SempreMilan",
    "user": {
        "utc_offset": -1,
        "friends_count": 4534,
        "listed_count": 5,
        "favourites_count": 3371,
        "verified": false,
        "description": "Imprenditore - Abbigliamento sportivo e da lavoro per tutte le attività - rivenditore all'ingrosso autorizzato Givova, Legea.\nPersonalizzazione stampe e ricami.",
        "created_at": {
            "$date": "2016-01-19T04:23:04Z"
        },
        "time_zone": null,
        "url": "http://www.abbigliamentosportdalavoro.it",
        "screen_name": "GianniLadage2",
        "statuses_count": 9613,
        "followers_count": 1085,
        "name": "Gianni Ladage",
        "location": "Milano, Lombardia",
        "id": 4826386474,
        "geo_enabled": true,
        "lang": null
    },
    "favorited": false
}
#### Optimized json
{
    "_id": {
        "$oid": "62165d624330c45866817a5f"
    },
    "in_reply_to_status_id": 1496379176995246083,
    "possibly_sensitive": false,
    "created_at": {
        "$date": "2022-02-23T16:13:53Z"
    },
    "source": "<a href=\"https://mobile.twitter.com\" rel=\"nofollow\">Twitter Web App</a>",
    "retweeted_status": {
        "id": 1496379176995246083,
        "user": {
           "friends_count": 4534,
           "listed_count": 5,
           "favourites_count": 3371,
           "statuses_count": 9613,
           "followers_count": 1085,
           "id": 4826386474,
        },
        "retweet_count": 226,
        "favourited_count": 325
    },
    "in_reply_to_screen_name": null,
    "in_reply_to_user_id": 626823042,
    "id": 1496518687930699780,
    "userMentionEntities": "JmsFortuna",
    "hashtagEntities": "MilanUdinese|SempreMilan",
    "user": {
        "friends_count": 4534,
        "listed_count": 5,
        "favourites_count": 3371,
        "verified": false,
        "created_at": {
            "$date": "2016-01-19T04:23:04Z"
        },
        "url": "http://www.abbigliamentosportdalavoro.it",
        "screen_name": "GianniLadage2",
        "statuses_count": 9613,
        "followers_count": 1085,
        "id": 4826386474,
        "geo_enabled": true,
    },
    "favorited": false
}

#### Compressed optimised json
[
   <tweet_id>,
   <in_reply_to_status_id>,
   <in_reply_to_user_id>,
   <possibly_sensitive>,
   <created_at>,
   <source>,
   <userMentionEntities>,
   <hashtagEntities>,
   [
      <user_id>
      <user_friends_count>,
      <user_followers_count>,
      <user_listed_count>,
      <user_favourites_count>,
      <user_verified>,
      <user_created_at>,
      <user_screen_name>,
      <user_has_url>,
      <user_statutes_count>,
      <user_geo_enabled>
   ],
   [
      <rt_id>,
      <rt_retweet_count>,
      <rt_favourited_count>,
      [
         <id>,
         <friends_count>,
         <listed_count>,
         <favourites_count>,
         <statuses_count>,
         <followers_count>,
      ]
   ],
]

[
   1496518687930699780,
   1496379176995246083,
   626823042,
   false,
   "2022-02-23T16:13:53Z",
   Twitter Web App,
   "JmsFortuna",
   "MilanUdinese|SempreMilan",
   [
      4826386474,
      4534,
      1085,
      5,
      3371,
      false,
      "2016-01-19T04:23:04Z",
      "GianniLadage2",
      true
      9613,
      true,
   ],
   [
      1496379176995246083,
      226,
      325
      [
         4826386474,
         4534,
         5,
         3371,
         9613,
         1085,
      ],
   ],
]

#### Compressed optimised json