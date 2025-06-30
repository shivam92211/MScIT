# Demonstrate the use of the Bitcoin Core API to interact with a Bitcoin Core node.


import requests

# Task 1: Get information regarding the current block
def get_current_block_info():
    response = requests.get("https://blockchain.info/latestblock")
    if response.status_code == 200:
        block_info = response.json()
        print("Current block information:")
        print("Block height:", block_info['height'])
        print("Block hash:", block_info['hash'])
        print("Block index:", block_info['block_index'])
        print("Timestamp:", block_info['time'])
    else:
        print("Failed to fetch block information")

# Task 2: Get balance of an address
def get_address_balance(address):
    response = requests.get(f"https://blockchain.info/q/addressbalance/{address}")
    if response.status_code == 200:
        balance = float(response.text) / 10**8
        print(f"Balance of address {address} : {balance} BTC")
    else:
        print("Failed to fetch balance for address:", address)

# Example usage
if __name__ == "__main__":
    get_current_block_info()
    address = "3Dh2ft6UsqjbTNzs5zrp7uK17Gqg1Pg5u5"
    get_address_balance(address)


