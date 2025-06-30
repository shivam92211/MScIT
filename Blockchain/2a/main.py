# Write a python program to demonstrate mining

import hashlib

def mine(block_data, difficulty=4):
    prefix = '0' * difficulty
    nonce = 0
    while True:
        text = block_data + str(nonce)
        hash_result = hashlib.sha256(text.encode()).hexdigest()
        if hash_result.startswith(prefix):
            print(f"Mined! Nonce: {nonce}, Hash: {hash_result}")
            break
        nonce += 1

mine("Hii You Mined a new block!")

