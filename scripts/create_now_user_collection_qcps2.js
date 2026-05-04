// Usage: mongosh "mongodb://<user>@127.0.0.1:27020/twitter?authSource=admin" -f create_now_user_collection_qcps2.js

const fs = require('fs');
const readline = require('readline');

// Ensure the target collection has a unique index on the user ID
db.QCPS_2_users.createIndex({ "id": 1 }, { unique: true });

async function processBatch(ids) {
    // Find tweets for this batch of IDs, sort by newest first, 
    // and extract the latest "user" object for each ID.
    const results = await db.QCPS_2.aggregate([
        { $match: { "user.id": { $in: ids } } },
        { $sort: { "created_at": -1 } },
        { $group: {
            _id: "$user.id",
            latest_user: { $first: "$user" }
        }}
    ], { allowDiskUse: true }).toArray();

    if (results.length === 0) return;

    // Prepare bulk insert/update operations (UPSERT)
    let bulkOps = results.map(doc => ({
        updateOne: {
            filter: { id: doc.latest_user.id },
            update: { $set: doc.latest_user },
            upsert: true
        }
    }));

    // ordered: false prevents the entire batch from failing if one document throws an error
    await db.QCPS_2_users.bulkWrite(bulkOps, { ordered: false });
}

async function run() {
    print("Starting extraction...");
    
    // Use a stream to avoid loading all 16M lines into RAM at once
    const fileStream = fs.createReadStream('all_tweet_authors_qcps2_safe.txt');
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    const BATCH_SIZE = 2000;
    let batchIds = [];
    let count = 0;

    for await (const line of rl) {
        let strId = line.trim();
        if (!strId) continue;

        // Convert to Long() so 64-bit Twitter IDs are not truncated by JS numbers
        batchIds.push(Long(strId));
        count++;

        if (batchIds.length >= BATCH_SIZE) {
            await processBatch(batchIds);
            batchIds = [];
            
            if (count % 100000 === 0) {
                print(`${count} IDs processed...`);
            }
        }
    }

    // Process any remaining IDs in the final batch
    if (batchIds.length > 0) {
        await processBatch(batchIds);
    }
    
    print(`Done! A total of ${count} IDs were processed.`);
}

run().catch(err => print("Error:", err));
