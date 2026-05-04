const fs = require('fs');

// 1. Définir le fichier d'entrée (REMPLACE PAR TON FICHIER DE 2000 USERS)
const inputFile = 'sample_2000_users.json';

let fileContent;
try {
    fileContent = fs.readFileSync(inputFile, 'utf8');
} catch (e) {
    console.error("Erreur: Impossible de lire le fichier " + inputFile);
    quit(1);
}

const ids = [];
const lines = fileContent.split('\n');

// 2. Extraire les IDs
for (let line of lines) {
    line = line.trim();
    if (!line || line === '[' || line === ']') continue;
    if (line.endsWith(',')) line = line.slice(0, -1);

    try {
        // Pour éviter la perte de précision fatale de Javascript sur les très grands nombres (Int64),
        // on extrait l'ID directement depuis le texte (Regex) sous forme de String, 
        // puis on le convertit en vrai Long() MongoDB.
        const idMatch = line.match(/"id"\s*:\s*(\d+)/);
        if (idMatch && idMatch[1]) {
            const idString = idMatch[1];
            ids.push(Long.fromString(idString));
        }
    } catch (e) {
        // Ignore les lignes mal formées
    }
}

// 3. Récupérer les textes par morceaux de 5000 pour ne pas saturer MongoDB
const batchSize = 5000;
for (let i = 0; i < ids.length; i += batchSize) {
    const batch = ids.slice(i, i + batchSize);

    const cursor = db.QCPS_2.find(
        { "id": { $in: batch } },
        { "_id": 0, "id": 1, "text": 1 }
    );

    while (cursor.hasNext()) {
        print(EJSON.stringify(cursor.next()));
    }
}
