# MCP Tools – Research and DB Query

Two FastMCP stdio servers:

## research_server.py
- Purpose: search arXiv and store results under `papers/<topic>/papers_info.json`.
- Install: `pip install arxiv mcp` (package name may be `fastmcp` in your env).
- Run: `python mcp/research_server.py` (connect with an MCP-compatible client).
- Tools:
  - `search_papers(topic, max_results=5)` → saves JSON and returns paper IDs.
  - `extract_info(paper_id)` → returns saved details as JSON string.
- Expected: prints the path where results are saved and returns IDs.

## db_query.py
- Purpose: query an Oracle table via `oracledb`.
- Install: `pip install oracledb mcp` and ensure Oracle client libs are configured.
- Env: `ORACLE_USER`, `ORACLE_PASSWORD`, `ORACLE_HOST`, `ORACLE_SERVICE` (and optional `ORACLE_PORT`).
- Run: `python mcp/db_query.py` (as an MCP stdio server). Use tool `query_user_table(where_clause="...")` from your client.
- Expected: returns list of rows; prints connection parameters for debugging.

