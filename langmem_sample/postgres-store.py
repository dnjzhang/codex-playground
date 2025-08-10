#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

# Use PostgresStore instead of InMemoryStore
from langgraph.store.postgres import PostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Connect to a PostgreSQL database (make sure it's running and accessible)
conn_string = "postgresql://langmem_app:john2025@localhost:5432/langmem_db"

# Initialize the Postgres-backed store with vector index configuration
with PostgresStore.from_conn_string(conn_string, index={
    "dims": 1536,                              # embedding dimensions
    "embed": "openai:text-embedding-3-small",  # embedding model for vector search
}) as store:
    store.setup()  # Run migrations to create tables and indexes (do this once)

    # Create memory tools with a namespace for emails (e.g., per user)
    manage_memory_tool = create_manage_memory_tool(
        namespace=("email_assistant", "user1", "collection"),
        store=store
    )
    search_memory_tool = create_search_memory_tool(
        namespace=("email_assistant", "user1", "collection"),
        store=store
    )

    email1 = "Reminder: Dinner will start at 6:00 PM sharp."
    email2 = "Project Update – Completed the initial design draft."

    # Store the emails in memory (create new memory entries)
    r1 = manage_memory_tool.invoke({"action": "create", "content": email1})
    print("create #1 ->", r1)

    r2 = manage_memory_tool.invoke({"action": "create", "content": email2})
    print("create #2 ->", r2)

    # Search memories (semantic search by query meaning)
    search_out = search_memory_tool.invoke({"query": "dinner", "limit": 5})
    print("\nsearch ->", search_out)
