#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

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


# Find memories
memories = store.search(("user_123", "memories"), query="everything", limit=30)
for memory in memories:
    print(f"Memory: {memory.value['text']} (similarity: {memory.score})")
    print("---")

# Fetch a memory entry with key
memory = store.get(("user_123", "memories"), "5")
print("fetch memory with id =5")
print(f"Memory: {memory}")

