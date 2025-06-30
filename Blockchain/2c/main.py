# Demonstrating the process of running a blockchain node on your local machine.

import hashlib
import json
from datetime import datetime


class Block:
    def __init__(self, index, timestamp, data, previous_hash=""):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        value = str(self.index) + self.previous_hash + self.timestamp + json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(value.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "09/06/2024", "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True


# Simulating a blockchain node
if __name__ == "__main__":
    my_coin = Blockchain()
    my_coin.add_block(Block(1, "09/06/2024", {"amount": 4}))
    my_coin.add_block(Block(2, "09/06/2024", {"amount": 8}))

    # Print blockchain
    print(json.dumps([block.__dict__ for block in my_coin.chain], indent=4))

    # Check if blockchain is valid
    print("Is blockchain valid?", my_coin.is_chain_valid())
