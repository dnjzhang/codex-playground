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
def query_user_table(where_clause: str = "") -> List[tuple]:
    """Query the Oracle users table.

    Args:
        where_clause: Optional SQL where clause (without the word WHERE).

    Returns:
        List of rows returned from the query.
    """

    # Determine connection parameters from environment variables
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    host = os.getenv("ORACLE_HOST")
    port = os.getenv("ORACLE_PORT", "1521")
    service = os.getenv("ORACLE_SERVICE")

    if not all([user, password, host, service]):
        raise ValueError(
            "Missing Oracle DB connection environment variables."
        )

    dsn = oracledb.makedsn(host, port, service_name=service)
    connection = oracledb.connect(user=user, password=password, dsn=dsn)
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

