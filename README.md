# gds-agent-benchmarks

### Make sure to only run the benchmarks in containerized or any other safe environment. The benchmark script contains `claude -p --dangerously-skip-permissions`, which means the LLM may decide to modify the system, including editting or deleting other files.

The MCP server of GDS agent is at: https://github.com/neo4j-contrib/gds-agent

Preparations:
1. Download a `gds_agent-0.5.1-py3-none-any.whl` wheel file which contains implementation of the MCP server.
2. Start a Neo4j database with the either LN or GoT dataset loaded.
3. Set your NEO4J configuration, e.g `NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD` in `benchmark_gds_agent.py`

Running benchamrks:
1. Run `python benchmark_gds_agent.py $dataset --model $model` to produce answers for the set of questions.
   Add `--with-cypher-mcp` to also start the read-only [`mcp-neo4j-cypher`](https://github.com/neo4j-contrib/mcp-neo4j) server next to gds-agent (same pairing as the gds-agent MCP config). Default is gds-agent only.
2. Run `python evaluate_benchmark.py $dataset --model $model` to use the produced answers and evaluate them.
3. Run `python analyze_benchmark_stats.py $dataset --model $model` to calculate any further summary statistics and plots.