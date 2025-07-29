#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

def parse_args():
    parser = argparse.ArgumentParser(description='Memory store search and retrieval')
    parser.add_argument('--query', '-q', 
                       help='Search query for finding similar memories',
                       default=None)
    parser.add_argument('--key', '-k',
                       help='Specific memory key to retrieve',
                       default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up vector store for similarity search
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    # Store some memories
    store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
    store.put(("user_123", "memories"), "2", {"text": "I prefer Italian food"})
    store.put(("user_123", "memories"), "3", {"text": "I don't like spicy food"})
    store.put(("user_123", "memories"), "4", {"text": "My major is in finance."})
    store.put(("user_123", "memories"), "5", {"text": "I am a plumber"})
    store.put(("user_123", "memories"), "6", {"text": "Looking for highest CD rate"})

    # Search by query if provided
    if args.query:
        print(f"\nSearching memories with query: '{args.query}'")
        memories = store.search(("user_123", "memories"), query=args.query, limit=3)
        for memory in memories:
            print(f"Memory: {memory.value['text']} (similarity: {memory.score})")
            print("---")

    # Fetch specific memory if key provided
    if args.key:
        print(f"\nFetching memory with key: '{args.key}'")
        memory = store.get(("user_123", "memories"), args.key)
        print(f"Memory: {memory}")

    # If no arguments provided, show usage
    if not (args.query or args.key):
        print("Please provide either --query (-q) or --key (-k) argument.")
        print("Example usage:")
        print("  python memory_store.py --query 'food'")
        print("  python memory_store.py --key '5'")

if __name__ == "__main__":
    main()