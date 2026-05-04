// Usage: mongosh "mongodb://<user>@127.0.0.1:27020/twitter?authSource=admin" -f count_how_many_users_exists_among_registry.js

let total = db.QCPS_2_accountRegistry.countDocuments();
let processed = 0;
let countFound = 0;
let batch = [];
let batchSize = 5000; // We verify 5000 users at once
let startTime = new Date();

print("Début du traitement pour " + total + " utilisateurs...");

// TIP : replace { id: 1 } and account.id with _id if your registry IDs are on _id
db.QCPS_2_accountRegistry.find({}, { id: 1 }).forEach(function(account) {
    if (account.id) { 
        batch.push(account.id);
    }
    processed++;

    // When we reach 5000, or when we are at the very last document
    if (batch.length === batchSize || processed === total) {
        
        // We are searching for all IDs in this batch that exist in QCPS_2
        let foundInBatch = db.QCPS_2.distinct("user.id", { "user.id": { $in: batch } });
        countFound += foundInBatch.length;
        
        // Progression bar display
        let elapsedSec = Math.max((new Date() - startTime) / 1000, 1);
        let speed = processed / elapsedSec;
        let percent = ((processed / total) * 100).toFixed(2);
        
        // process.stdout.write avec "\r" allows to erase the previous line !
        process.stdout.write("\r🔄 Processed: " + processed + " / " + total + " (" + percent + "%) | Found: " + countFound + " | Speed: " + Math.round(speed) + " req/sec   ");
        
        // We empty the batch for the next 5000
        batch = [];
    }
});

print("\n✅ --- FINISHED ---");
print("Exact number of users from the registry who have posted in QCPS_2: " + countFound);
