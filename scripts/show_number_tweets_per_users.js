// Usage: mongosh "mongodb://<user>@127.0.0.1:27020/twitter?authSource=admin" -f show_number_tweets_per_users.js

// We define how many users we want to analyze for this test
const MAX_USERS_TO_DISPLAY = 50; 

print("🚀 Starting stream (Sorting by user.id)...");

// We open a sorted cursor
const cursor = db.QCPS_2.find({}, { "user.id": 1, "user.screen_name": 1 })
                        .sort({ "user.id": 1 });

let currentUser = { id: null, name: "", count: 0 };
let usersFound = 0;

while (cursor.hasNext() && usersFound < MAX_USERS_TO_DISPLAY) {
    const tweet = cursor.next();
    const userId = tweet.user.id;

    // We detect the sequence break (User change)
    if (userId !== currentUser.id) {
        if (currentUser.id !== null) {
            print(`User ID: ${currentUser.id} | Name: @${currentUser.name.padEnd(15)} | Tweets: ${currentUser.count}`);
            usersFound++;
        }

        // Reset for the new user
        currentUser = {
            id: userId,
            name: tweet.user.screen_name || "N/A",
            count: 1
        };
    } else {
        // Same user : we increment the counter
        currentUser.count++;
    }
}

// Display the last processed user
if (currentUser.id !== null && usersFound < MAX_USERS_TO_DISPLAY) {
    print(`User ID: ${currentUser.id} | Name: @${currentUser.name.padEnd(15)} | Tweets: ${currentUser.count}`);
}

print("\n✅ Preview completed.");