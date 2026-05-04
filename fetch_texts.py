import json
from pymongo import MongoClient

def get_id_value(val):
    """Extrait l'entier d'un champ id, peu importe comment il a été formaté dans le JSON"""
    if isinstance(val, dict):
        if "$numberLong" in val:
            return int(val["$numberLong"])
        if "low" in val and "high" in val:
            return (val["high"] << 32) + (val["low"] & 0xFFFFFFFF)
    return int(val)

def fetch_texts():
    input_file = "sample_100_users.json"  # REMPLACE par le nom de ton fichier de 2000 users !
    output_file = "tweet_texts.json"
    
    # 1. Lire le fichier local pour récupérer tous les IDs
    print(f"Lecture des IDs depuis {input_file}...")
    tweet_ids = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line == "[" or line == "]": 
                    continue
                if line.endswith(","): 
                    line = line[:-1]
                    
                try:
                    doc = json.loads(line)
                    if "id" in doc:
                        tweet_ids.append(get_id_value(doc["id"]))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Le fichier {input_file} est introuvable. Modifie la variable input_file dans le script.")
        return

    print(f"✅ {len(tweet_ids)} IDs de tweets trouvés.")
    
    if len(tweet_ids) == 0:
        return

    # 2. Connexion à MongoDB (ta base de Prod ou celle qui contient les textes)
    # Si ta prod est distante, change "localhost" par l'IP de ta prod
    client = MongoClient("mongodb://localhost:27017/") 
    db = client["twitter"]
    col = db["QCPS_2"]
    
    print(f"Récupération des textes depuis MongoDB...")
    
    batch_size = 5000
    fetched_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # On interroge MongoDB par paquets pour ne pas surcharger la RAM
        for i in range(0, len(tweet_ids), batch_size):
            batch = tweet_ids[i:i+batch_size]
            
            cursor = col.find(
                {"id": {"$in": batch}},
                {"_id": 0, "id": 1, "text": 1}
            )
            
            for doc in cursor:
                out_doc = {
                    "id": get_id_value(doc["id"]),
                    "text": doc.get("text", "")
                }
                # Sauvegarde au format JSON Lines (un objet par ligne)
                out_f.write(json.dumps(out_doc, ensure_ascii=False) + "\n")
                fetched_count += 1
                
            print(f"... {fetched_count} textes récupérés ...")
            
    print(f"\n🎉 Terminé ! {fetched_count} textes ont été sauvegardés dans '{output_file}'.")

if __name__ == "__main__":
    fetch_texts()
