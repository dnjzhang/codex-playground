#!/usr/bin/env python3
import argparse
from typing import Optional
import sys
from letta_client import Letta
from tabulate import tabulate
import json


class ArchivalMemoryManager:
    def __init__(self, base_url: str = "http://localhost:8283"):
        self.client = Letta(base_url=base_url)

    def list_sources(self) -> None:
        """List all available sources in archival memory"""
        try:
            sources = self.client.sources.list()
            if not sources:
                print("No sources found in archival memory")
                return

            # Print raw source object for debugging
            #print("Debug - First source object attributes:", vars(sources[0]))

            # Prepare table data
            table_data = []
            for source in sources:
                try:
                    num_passages = len(self.client.sources.passages.list(source_id=source.id))
                    # Only include attributes that we know exist
                    table_data.append([
                        source.id,
                        source.name,
                        num_passages
                    ])
                except Exception as inner_e:
                    import traceback
                    print(f"Error processing source {source.id}:")
                    print(traceback.format_exc())

            # Print formatted table
            headers = ["ID", "Name", "Passages"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        except Exception as e:
            import traceback
            print("Error listing sources:")
            print(traceback.format_exc())

    def show_source_metadata(self, source_id: str) -> None:
        """Show detailed metadata for a specific source"""
        try:
            # Get source details
            source = self.client.sources.retrieve(source_id)

            # Get passages
            passages = self.client.sources.passages.list(source_id=source_id)

            # Print source information using safe attribute access
            print("\n=== Source Information ===")
            print(f"ID: {getattr(source, 'id', 'N/A')}")
            print(f"Name: {getattr(source, 'name', 'N/A')}")
            print(f"Total Passages: {len(passages)}")

            # Print all available attributes for debugging
            print("\nAll available attributes:")
            for attr_name in dir(source):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(source, attr_name)
                        if not callable(value):  # Skip methods
                            print(f"{attr_name}: {value}")
                    except Exception:
                        continue

            # Print sample passages (first 3)
            # if passages:
            #     print("\n=== Sample Passages ===")
            #     for i, passage in enumerate(passages[:3]):
            #         print(f"\nPassage {i + 1}:")
            #         try:
            #             if hasattr(passage, 'text'):
            #                 print(f"Text: {passage.text[:200]}...")
            #             if hasattr(passage, 'metadata'):
            #                 print(f"Metadata: {json.dumps(passage.metadata, indent=2)}")
            #         except Exception as passage_e:
            #             print(f"Error processing passage: {passage_e}")

        except Exception as e:
            import traceback
            print("\nError showing source metadata:")
            print(traceback.format_exc())

    def delete_source(self, source_id: str) -> None:
        """Delete a source and all its passages"""
        try:
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete source {source_id}? (y/N): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled")
                return

            self.client.sources.delete(source_id)
            print(f"Successfully deleted source {source_id}")

        except Exception as e:
            print(f"Error deleting source: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Letta Archival Memory Manager")
    parser.add_argument("--url", default="http://localhost:8283", help="Letta server URL")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all sources in archival memory")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show source metadata")
    show_parser.add_argument("source_id", help="ID of the source to show")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a source")
    delete_parser.add_argument("source_id", help="ID of the source to delete")

    args = parser.parse_args()

    manager = ArchivalMemoryManager(args.url)

    if args.command == "list":
        manager.list_sources()
    elif args.command == "show":
        manager.show_source_metadata(args.source_id)
    elif args.command == "delete":
        manager.delete_source(args.source_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()