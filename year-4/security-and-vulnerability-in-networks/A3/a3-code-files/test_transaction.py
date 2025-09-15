import unittest
import binascii
from transaction import Transaction
from wallet import Wallet
from block import Block
from blockchain import Blockchain


# DO NOT CHANGE THIS CODE
class TestTransaction(unittest.TestCase):

    def setUp(self):
        self.sender_wallet_address = "09dc52e95eddfc988db84010c702d30f75fc37674ff6441462ccf62f49122cbbd1336945012fe1c6d994cc30150e8e9844484a2d0f76b8979a489e28985b50ab"
        self.recipient_wallet_address = "f4c20f6e881d621da6663eaa15d0ee70667f1020e7aea75fb58e2683efae8242cca50f173780c8feba3b8e0a855789a24ade7c17558f96bbd65a1d5c2be75561"
        self.amount = 10
        self.sender_wallet_public_key = "eb9df3870f53afb93ae6e07a44512ff05bcbf12f6e776872f60c8460e772d672aea7631f3ca25b5fffff63935eac2736941ad20f6115a28f40820d763c236172"
        self.sender_wallet_private_key = (
            "38a52d4024ea15c38972ab0bc67a926a7c57f5b64983064eaa023943fc5f6efe"
        )

        self.transaction = Transaction(
            sender_address=self.sender_wallet_address,
            recipient_address=self.recipient_wallet_address,
            amount=self.amount,
            sender_public_key=self.sender_wallet_public_key,
        )

    def test_compute_hash(self):

        tx_hash = self.transaction.compute_hash()

        self.assertIsInstance(tx_hash, str)
        self.assertEqual(len(tx_hash), 64)
        self.assertEqual(
            tx_hash, "8acaafa1d8232aaf065e5b6e30fc81d2ab97a41a7c2b31d7bc024579d68943b0"
        )
        # Ensure that the hash is consistent
        tx_hash_again = self.transaction.compute_hash()
        self.assertEqual(tx_hash, tx_hash_again)

    def test_sign_transaction(self):
        self.transaction.sign_transaction(self.sender_wallet_private_key)

        self.assertIsNotNone(self.transaction.signature)

        # Signature should be a hex string
        self.assertIsInstance(self.transaction.signature, str)
        self.assertEqual(
            self.transaction.signature,
            "6b862726ce00e63b0cde658b16d17918d3f6d16f96f6b262978ffcc7cc90857acde092625f51067741a8a6e5f1f93e6c71f7b8b893e3207c3a1ea5485ae4aa5d",
        )

    def test_is_valid(self):
        # Test with a valid signature
        self.transaction.sign_transaction(self.sender_wallet_private_key)
        self.assertTrue(self.transaction.is_valid())

        # Test with an invalid signature
        invalid_signature = "a" * len(self.transaction.signature)
        self.transaction.signature = invalid_signature
        self.assertFalse(self.transaction.is_valid())

        # Test without a signature
        self.transaction.signature = None
        self.assertFalse(self.transaction.is_valid())


if __name__ == "__main__":
    unittest.main()
