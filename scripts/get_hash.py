import sys
import mmh3

_B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def int_to_base62(n):
    n = int(n) & 0xFFFFFFFFFFFFFFFF  # Force le 64-bit non signé
    chars = []
    while n:
        chars.append(_B62[n % 62])
        n //= 62
    return ''.join(reversed(chars)) if chars else '0'

def get_node_id(user_id):
    # 1. Calcul du hash (non signé) via MurmurHash3
    hash_value = mmh3.hash64(str(user_id), 0, signed=False)[0]
    
    # 2. Conversion en Base-62 du hash
    node_id_hashed = int_to_base62(hash_value)
    return hash_value, node_id_hashed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilisation : python get_hash.py <id_utilisateur>")
        sys.exit(1)
        
    user_id = sys.argv[1]
    
    # Calcul avec Hash
    hash_val, node_id_hashed = get_node_id(user_id)
    
    # Calcul sans Hash (ID original directement en base62)
    try:
        node_id_not_hashed = int_to_base62(int(user_id))
    except ValueError:
        node_id_not_hashed = "Erreur (L'ID original n'est pas un entier valide)"
    
    print(f"ID Original          : {user_id}")
    print(f"Hash 64-bit          : {hash_val}")
    print(f"Base62 (Hashed)      : {node_id_hashed}")
    print(f"Base62 (Not hashed)  : {node_id_not_hashed}")
