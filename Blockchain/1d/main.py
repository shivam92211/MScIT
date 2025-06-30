# Implement a function to add new blocks to the miner and dump the blockchain



import datetime
import hashlib

class Block:
    def __init__(self, data, previous_hash):
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calc_hash()

    def calc_hash(self):
        sha = hashlib.sha256()
        hash_str = self.data.encode("utf-8")
        sha.update(hash_str)
        return sha.hexdigest()


if __name__ == "__main__":
    # Create a simple blockchain with 3 blocks
    blockchain = [Block("First block", "0")]
    blockchain.append(Block("Second block", blockchain[0].hash))
    blockchain.append(Block("Third block", blockchain[1].hash))

    # Print the blockchain
    for block in blockchain:
        print(f"Timestamp: {block.timestamp}")
        print(f"Data: {block.data}")
        print(f"Previous Hash: {block.previous_hash}")
        print(f"Hash: {block.hash}")
        print("-" * 50)
