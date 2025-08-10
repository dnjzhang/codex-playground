#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

from langmem import create_manage_memory_tool, create_search_memory_tool
#%%

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create memory tools with a namespace for emails (e.g., per user)
manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "user1", "collection"),  # using user "user1"
    store=store  # provide store when using tools outside a LangGraph agent context:contentReference[oaicite:3]{index=3}
)
search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "user1", "collection"),
    store=store
)

email1 = "Reminder: Team meeting tomorrow at 10 AM."
email2 = "Project Update – Completed the initial design draft."

# Store the emails in memory (create new memory entries)
r1 = manage_memory_tool.invoke({"action": "create", "content": email1})
print("create #1 ->", r1)

r2 = manage_memory_tool.invoke({"action": "create", "content": email2})
print("create #2 ->", r2)

# 4) Search memories (semantic)
search_out = search_memory_tool.invoke({"query": "meeting", "limit": 5})
print("\nsearch ->", search_out)

