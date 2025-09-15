import threading
import time
from queue import Queue, Empty
from blockchain import Blockchain
from wallet import Wallet

# Initialize the blockchain with a transaction limit
max_tx_per_block = 10  # Maximum transactions per block
difficulty = 3  # Set a low difficulty
delay = 0.02
transaction_queue = Queue()
blockchain = Blockchain(
    max_transactions_per_block=max_tx_per_block, difficulty=difficulty
)

# Create wallets for users
users = {}
for i in range(10):
    user_id = f"User_{i}"
    users[user_id] = Wallet()


# Function to continuously generate transactions
def generate_transactions():
    transaction_count = 0
    while transaction_count < 100:  # Limit to 100 transactions for this test
        sender_id = f"User_{transaction_count % 10}"
        recipient_id = f"User_{(transaction_count + 1) % 10}"
        amount = (transaction_count + 1) * 0.1
        sender_wallet = users[sender_id]
        recipient_wallet = users[recipient_id]

        transaction = sender_wallet.create_transaction(recipient_wallet.address, amount)
        # Put the transaction into the queue
        transaction_queue.put(transaction)
        print(f"Transaction {transaction_count + 1} added: {transaction}")

        transaction_count += 1
        time.sleep(delay)  # Pause briefly to simulate real-world delays


# Function to perform mining
def mine_blockchain():
    while True:
        transactions_to_mine = []
        # Fetch all available transactions from the queue up to the max per block
        while (
            not transaction_queue.empty()
            and len(transactions_to_mine) < blockchain.max_transactions_per_block
        ):
            try:
                transaction = transaction_queue.get_nowait()
                transactions_to_mine.append(transaction)
            except Empty:
                break  # No more transactions available

        # Even if there are fewer transactions, proceed to mine
        if transactions_to_mine:
            # Set the unconfirmed transactions to the collected transactions
            blockchain.unconfirmed_transactions = transactions_to_mine
            print("Mining started...")
            block_index = blockchain.mine()
            if block_index:
                print(
                    f"Mining completed. Block {block_index} has been added to the blockchain with {len(transactions_to_mine)} transaction(s)."
                )
            else:
                print("Mining failed.")
        else:
            # No transactions to mine at the moment
            time.sleep(delay)  # Wait a bit before trying again

        # For demonstration purposes, exit when both threads are done
        if not transaction_thread.is_alive() and transaction_queue.empty():
            print("No more transactions to process. Mining thread is exiting.")
            break


# Create threads for transaction generation and mining
transaction_thread = threading.Thread(target=generate_transactions)
mining_thread = threading.Thread(target=mine_blockchain)

# Start the threads
transaction_thread.start()
mining_thread.start()

# Wait for both threads to finish
transaction_thread.join()
mining_thread.join()

# Print the blockchain summary after all transactions are added
print("\nBlockchain Summary:")
print(blockchain)
