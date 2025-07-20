import os
from typing import List

# Import the same FastMCP server used by research_server
from mcp.server.fastmcp import FastMCP
import oracledb

# Initialize FastMCP server for the database query scenario
mcp = FastMCP("db_query")

# Add a prompt describing how to query the user table
mcp.prompt = (
    "Use the 'query_user_table' tool to retrieve rows from the Oracle 'users' table. "
    "Provide optional SQL conditions as a where_clause argument to filter results."
)

# Expose the repository README file as a resource
@mcp.resource()
def readme() -> str:
    """Return the contents of the repository README file."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path, "r") as f:
        return f.read()

@mcp.tool()
def query_user_table(connection_str: str, where_clause: str = "") -> List[tuple]:
    """Query the Oracle users table.

    Args:
        connection_str: Oracle connection string in the form
            'user/password@host:port/service'.
        where_clause: Optional SQL where clause (without the word WHERE).

    Returns:
        List of rows returned from the query.
    """
    connection = oracledb.connect(connection_str)
    cursor = connection.cursor()

    query = "SELECT * FROM user_def"
    if where_clause:
        query += f" WHERE {where_clause}"
    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    connection.close()
    return rows

if __name__ == "__main__":
    mcp.run(transport="stdio")

