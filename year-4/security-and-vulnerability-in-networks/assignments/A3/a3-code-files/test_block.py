import unittest
import binascii
from transaction import Transaction
from wallet import Wallet
from block import Block
from blockchain import Blockchain


# DO NOT CHANGE THIS CODE
class TestBlock(unittest.TestCase):

    def test_compute_hash(self):
        block = Block(index=1, transactions=[], previous_hash="0")
        block.timestamp = 1727951150
        block_hash = block.compute_hash()

        self.assertIsInstance(block_hash, str)
        self.assertEqual(len(block_hash), 64)
        self.assertEqual(
            block_hash,
            "663277ad072c15ade3856afcd9fa0eca0391b597e3915a2e4a8877b98ceb4d89",
        )
        # Ensure that the hash is consistent
        block_hash_again = block.compute_hash()
        self.assertEqual(block_hash, block_hash_again)


if __name__ == "__main__":
    unittest.main()
