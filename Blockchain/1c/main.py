# Create a Python class named Transaction with attributes for sender, receiver, and amount.
# Implement a method within the class to transfer money from the sender's account to the
# receiver's account.



from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA
import Crypto
import datetime, collections, binascii


class Client:
    def __init__(self):
        random = Crypto.Random.new().read
        self._private_key = RSA.generate(1024, random)
        self._public_key = self._private_key.publickey()

    @property
    def identity(self):
        return binascii.hexlify(
            self._public_key.exportKey(format="DER")
        ).decode("ascii")


class Transaction:
    def __init__(self, sender, receiver, value):
        self.sender = sender
        self.receiver = receiver
        self.value = value
        self.time = datetime.datetime.now()

    def to_dict(self):
        identity = "Genesis" if self.sender == "Genesis" else self.sender.identity
        return collections.OrderedDict({
            "sender": identity,
            "receiver": self.receiver,
            "value": self.value,
            "time": self.time.isoformat(),
        })

    def sign_transaction(self):
        signer = PKCS1_v1_5.new(self.sender._private_key)
        h = SHA.new(str(self.to_dict()).encode("utf8"))
        return binascii.hexlify(signer.sign(h)).decode("ascii")


def display_transaction(tx):
    d = tx.to_dict()
    print(f"Sender: {d['sender']}\nReceiver: {d['receiver']}\nValue: {d['value']}\nTime: {d['time']}")
    print("-" * 40)


# Sample clients
ninad = Client()
ks = Client()
vighnesh = Client()
sairaj = Client()

# Create and sign transactions
transactions = [
    Transaction(ninad, ks.identity, 15.0),
    Transaction(ninad, vighnesh.identity, 6.0),
    Transaction(ninad, sairaj.identity, 16.0),
    Transaction(vighnesh, ninad.identity, 8.0),
    Transaction(vighnesh, ks.identity, 19.0),
    Transaction(vighnesh, sairaj.identity, 35.0),
    Transaction(sairaj, vighnesh.identity, 5.0),
    Transaction(sairaj, ninad.identity, 12.0),
    Transaction(sairaj, ks.identity, 25.0),
    Transaction(ninad, ks.identity, 1.0),
]

for t in transactions:
    t.sign_transaction()
    display_transaction(t)
