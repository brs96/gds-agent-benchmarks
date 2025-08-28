# gds-agent-benchmarks

### Make sure to only run the benchmarks in containerized or any other safe environment. The benchmark script contains `claude -p --dangerously-skip-permissions`, which means the LLM may decide to modify the system, including editting or deleting other files.

The MCP server of GDS agent is at: https://github.com/neo4j-contrib/gds-agent/pulls

Preparations:
1. Download a `gds_agent-0.3.0-py3-none-any.whl` wheel file which contains implementation of the MCP server.
2. Start a Neo4j database with the London underground map loaded.
3. Set your NEO4J configuration in `benchmark_gds_agent.py`

Running benchamrks:
1. Run `python benchmark_gds_agent.py` to produce answers for the set of questions.
2. Run `python evaluate_benchmark.py` to use the produced answers and evaluate them.
3. Run `python analyze_benchmark_stats.py` to calculate any further summary statistics and plots.