#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# Set up vector store for similarity search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
class Episode(BaseModel):
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the memory, saving the agent's key internal thought process so it can learn over time."""

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )


# Configure memory manager with storage
manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("memories", "episodes"),
    schemas=[Episode],
    instructions="Extract exceptional examples of noteworthy problem-solving scenarios, including what made them effective.",
    enable_inserts=True,
)

llm = init_chat_model("openai:gpt-4o-mini")


@entrypoint(store=store)
def app(messages: list):
    # Step 1: Find similar past episodes
    similar = store.search(
        ("memories", "episodes"),
        query=messages[-1]["content"],
        limit=1,
    )

    # Step 2: Build system message with relevant experience
    system_message = "You are a helpful assistant."
    if similar:
        system_message += "\n\n### EPISODIC MEMORY:"
        for i, item in enumerate(similar, start=1):
            episode = item.value["content"]
            system_message += f"""

Episode {i}:
When: {episode['observation']}
Thought: {episode['thoughts']}
Did: {episode['action']}
Result: {episode['result']}
        """

    # Step 3: Generate response using past experience
    response = llm.invoke([{"role": "system", "content": system_message}, *messages])

    # Step 4: Store this interaction if successful
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "What's a binary tree? I work with family trees if that helps",
        },
    ],
)
print(store.search(("memories", "episodes"), query="Trees"))

# [
#     Item(
#         namespace=["memories", "episodes"],
#         key="57f6005b-00f3-4f81-b384-961cb6e6bf97",
#         value={
#             "kind": "Episode",
#             "content": {
#                 "observation": "User asked about binary trees and mentioned familiarity with family trees. This presented an opportunity to explain a technical concept using a relatable analogy.",
#                 "thoughts": "I recognized this as an excellent opportunity to bridge understanding by connecting a computer science concept (binary trees) to something the user already knows (family trees). The key was to use their existing mental model of hierarchical relationships in families to explain binary tree structures.",
#                 "action": "Used family tree analogy to explain binary trees: Each person (node) in a binary tree can have at most two children (left and right), unlike family trees where people can have multiple children. Drew parallel between parent-child relationships in both structures while highlighting the key difference of the two-child limitation in binary trees.",
#                 "result": "Successfully translated a technical computer science concept into familiar terms. This approach demonstrated effective teaching through analogical reasoning - taking advantage of existing knowledge structures to build new understanding. For future similar scenarios, this reinforces the value of finding relatable real-world analogies when explaining technical concepts. The family tree comparison was particularly effective because it maintained the core concept of hierarchical relationships while clearly highlighting the key distinguishing feature (binary limitation).",
#             },
#         },
#         created_at="2025-02-09T03:40:11.832614+00:00",
#         updated_at="2025-02-09T03:40:11.832624+00:00",
#         score=0.30178054939692683,
#     )
# ]