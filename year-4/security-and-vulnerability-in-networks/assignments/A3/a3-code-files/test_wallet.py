import unittest
import binascii
from transaction import Transaction
from wallet import Wallet
from block import Block
from blockchain import Blockchain


# DO NOT CHANGE THIS CODE
class TestWallet(unittest.TestCase):

    def test_generate_private_key(self):
        private_key = Wallet.generate_private_key()
        self.assertIsInstance(private_key, str)
        self.assertEqual(len(private_key), 64)
        # Ensure that the private key is hex
        int(private_key, 16)

    def test_get_public_key(self):
        private_key = Wallet.generate_private_key()
        public_key = Wallet.get_public_key(private_key)
        self.assertIsInstance(public_key, str)
        self.assertEqual(len(public_key), 128)
        # Ensure that the public key is hex
        int(public_key, 16)

    def test_create_transaction(self):
        sender_wallet = Wallet()
        recipient_wallet = Wallet()

        amount = 15
        transaction = sender_wallet.create_transaction(
            recipient_address=recipient_wallet.address, amount=amount
        )
        self.assertIsInstance(transaction, Transaction)
        self.assertEqual(transaction.sender_address, sender_wallet.address)
        self.assertEqual(transaction.recipient_address, recipient_wallet.address)
        self.assertEqual(transaction.amount, amount)
        self.assertIsNotNone(transaction.signature)
        self.assertTrue(transaction.is_valid())


if __name__ == "__main__":
    unittest.main()
