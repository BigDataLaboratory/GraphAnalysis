// Run with: mongosh "mongodb://..." -f extract_all_user_id.js

const outputFile = './all_tweet_authors_qcps2_safe.txt';
const collectionName = "QCPS_2";
const BATCH_SIZE = 50000;

print("🚀 Starting SAFE author extraction via mongosh...");

// 1. Reset file
fs.writeFileSync(outputFile, "");

// 2. Optimized Pipeline
const pipeline = [
    { $project: { _id: 0, "user.id": 1 } },
    { $group: { _id: "$user.id" } }
];

const cursor = db.getCollection(collectionName).aggregate(pipeline, { 
    allowDiskUse: true, 
    cursor: { batchSize: BATCH_SIZE } 
});

print("⏳ Pipeline started. MongoDB is grouping unique IDs (this might take a while)...");

let processed = 0;
let writeBuffer = [];

// 3. Iteration
while (cursor.hasNext()) {
    const doc = cursor.next();
    if (doc._id) {
        writeBuffer.push(doc._id.toString());
    }
    
    processed++;
    
    if (writeBuffer.length >= BATCH_SIZE) {
        // Use writeFileSync with the flag 'a' for append
        fs.appendFileSync(outputFile, writeBuffer.join('\n') + '\n');
        writeBuffer = []; 
        print("✅ Unique authors found: " + processed.toLocaleString());
    }
}

// 4. Final Flush
if (writeBuffer.length > 0) {
    fs.appendFileSync(outputFile, writeBuffer.join('\n') + '\n');
}

print("\n🎉 DONE! Total count: " + processed.toLocaleString());
