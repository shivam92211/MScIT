# Develop a secure messaging application where users can exchange messages securely using
# RSA encryption. Implement a mechanism for generating RSA key pairs and
# encrypting/decrypting messages.

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def generate_rsa_key_pair():
    """Generates a new RSA public and private key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key


def encrypt_message(public_key, message):
    """Encrypts a message using the recipient's public key."""
    ciphertext = public_key.encrypt(
        message.encode('utf-8'),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext


def decrypt_message(private_key, ciphertext):
    """Decrypts a ciphertext using the recipient's private key."""
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext.decode('utf-8')


if __name__ == "__main__":
    # User 1 generates their keys
    print("User 1: Generating RSA key pair...")
    user1_private_key, user1_public_key = generate_rsa_key_pair()
    print("User 1: Key pair generated.")

    # User 2 generates their keys
    print("\nUser 2: Generating RSA key pair...")
    user2_private_key, user2_public_key = generate_rsa_key_pair()
    print("User 2: Key pair generated.")

    # User 1 sends a message to User 2
    original_message_user1 = "Hello User 2, this is a secret message from User 1!"
    print(f"\nUser 1: Original message to User 2: '{original_message_user1}'")

    # Encrypt with User 2's public key
    encrypted_message_user1_to_user2 = encrypt_message(user2_public_key, original_message_user1)
    print(f"User 1: Encrypted message (ciphertext): {encrypted_message_user1_to_user2}")

    # Decrypt with User 2's private key
    decrypted_message_user2 = decrypt_message(user2_private_key, encrypted_message_user1_to_user2)
    print(f"User 2: Decrypted message: '{decrypted_message_user2}'")

    # User 2 sends a reply to User 1
    original_message_user2 = "Hi User 1, I received your message securely!"
    print(f"\nUser 2: Original message to User 1: '{original_message_user2}'")

    # Encrypt with User 1's public key
    encrypted_message_user2_to_user1 = encrypt_message(user1_public_key, original_message_user2)
    print(f"User 2: Encrypted reply (ciphertext): {encrypted_message_user2_to_user1}")

    # Decrypt with User 1's private key
    decrypted_message_user1 = decrypt_message(user1_private_key, encrypted_message_user2_to_user1)
    print(f"User 1: Decrypted reply: '{decrypted_message_user1}'")
