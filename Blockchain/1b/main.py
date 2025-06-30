# Allow users to create multiple transactions and display them in an organised format.

import Crypto
import binascii
import datetime
import collections

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA

class Client:
    def __init__(self):
        # Create random bytes for key generation
        random = Crypto.Random.new().read
        self._private_key = RSA.generate(1024, random)
        self._public_key = self._private_key.publickey()
        self._signer = PKCS1_v1_5.new(self._private_key)

    @property
    def identity(self):
        return binascii.hexlify(
            self._public_key.exportKey(format='DER')
        ).decode('ascii')


class Transaction:
    def __init__(self, sender, receiver, value):
        self.sender = sender
        self.receiver = receiver
        self.value = value
        self.time = datetime.datetime.now()

    def to_dict(self):
        if self.sender == "Genesis":
            identity = "Genesis"
        else:
            identity = self.sender.identity
        return collections.OrderedDict({
            'sender': identity,
            'receiver': self.receiver,
            'value': self.value,
            'time': self.time.isoformat()
        })

    def sign_transaction(self):
        private_key = self.sender._private_key
        signer = PKCS1_v1_5.new(private_key)
        h = SHA.new(str(self.to_dict()).encode('utf-8'))
        return binascii.hexlify(signer.sign(h)).decode('ascii')


# Example usage
Raj = Client()
print("-" * 50)
print("Raj Key")
print(Raj.identity)

Vai = Client()
print("-" * 50)
print("Vai Key")
print(Vai.identity)

t = Transaction(Raj, Vai.identity, 10.0)
print("-" * 50)
print("Transaction Dictionary")
print(t.to_dict())

print("-" * 50)
print("Transaction Signature")
signature = t.sign_transaction()
print(signature)
print("-" * 50)
