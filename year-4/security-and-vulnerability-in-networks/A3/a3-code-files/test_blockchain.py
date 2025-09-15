import unittest
import binascii
from transaction import Transaction
from wallet import Wallet
from block import Block
from blockchain import Blockchain


# DO NOT CHANGE THIS CODE
class TestBlockchain(unittest.TestCase):

    def setUp(self):
        self.blockchain = Blockchain(max_transactions_per_block=2)
        self.wallet1 = Wallet()
        self.wallet2 = Wallet()
        # Create a valid transaction
        self.transaction = self.wallet1.create_transaction(
            recipient_address=self.wallet2.address, amount=50
        )

    def test_proof_of_work(self):
        block = Block(index=1, transactions=[], previous_hash="0")
        proof = self.blockchain.proof_of_work(block)
        self.assertTrue(proof.startswith("0" * self.blockchain.difficulty))
        self.assertEqual(block.hash, proof)

    def test_add_block(self):
        block = Block(
            index=1,
            transactions=[self.transaction],
            previous_hash=self.blockchain.last_block.hash,
        )
        proof = self.blockchain.proof_of_work(block)
        added = self.blockchain.add_block(block, proof)
        self.assertTrue(added)
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(self.blockchain.last_block, block)

    def test_is_valid_proof(self):
        block = Block(index=1, transactions=[], previous_hash="0")
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith("0" * self.blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        valid = self.blockchain.is_valid_proof(block, computed_hash)
        self.assertTrue(valid)

    def test_add_transaction(self):
        added = self.blockchain.add_transaction(self.transaction)
        self.assertTrue(added)
        self.assertEqual(len(self.blockchain.unconfirmed_transactions), 1)

    def test_mine(self):
        self.blockchain.add_transaction(self.transaction)
        last_block_index = self.blockchain.mine()
        self.assertIsNotNone(last_block_index)
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(len(self.blockchain.unconfirmed_transactions), 0)

    def test_is_chain_valid(self):
        self.blockchain.add_transaction(self.transaction)
        self.blockchain.mine()
        valid = self.blockchain.is_chain_valid()
        self.assertTrue(valid)

        # Tamper with the blockchain
        self.blockchain.chain[1].transactions[0].amount = 1000
        valid = self.blockchain.is_chain_valid()
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
