# gds-agent-benchmarks

### Make sure to only run the benchmarks in containerized or any other safe environment. The benchmark script contains `claude -p --dangerously-skip-permissions`, which means the LLM may decide to modify the system, including editting or deleting other files.

The MCP server of GDS agent is at: [https://github.com/neo4j-contrib/gds-agent](https://github.com/neo4j-contrib/gds-agent)

Preparations:

1. Install the Python requirements and make sure `uvx` is available. The benchmark uses `gds-agent==1.0.1` from PyPI by default.
2. Start a Neo4j database with either the LN or GoT dataset loaded.
3. Put the database configuration in `.env` (or export it in the shell):
  ```dotenv
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-password
   # NEO4J_DATABASE=neo4j
  ```

For an AuraDB that uses Aura Graph Analytics sessions, also configure:

```dotenv
AURA_API_CLIENT_ID=your-client-id
AURA_API_CLIENT_SECRET=your-client-secret
# AURA_API_PROJECT_ID=your-project-id
# SESSION_MEMORY_GB=8
# SESSION_TTL_HOURS=24
```

`gds-agent` detects plugin mode or Aura session mode from the connected database. In session mode, the skill instructs the model to create and clean up sessions.

Running benchmarks:

1. Run `python benchmark_gds_agent.py $dataset --model $model` to produce answers for the set of questions.
  Add `--with-cypher-mcp` to also start the read-only [`mcp-neo4j-cypher`](https://github.com/neo4j-contrib/mcp-neo4j) server next to gds-agent (same pairing as the gds-agent MCP config). Default is gds-agent only.
2. Run `python evaluate_benchmark.py $dataset --model $model` to use the produced answers and evaluate them.
3. Run `python analyze_benchmark_stats.py $dataset --model $model` to calculate any further summary statistics and plots.

Use `--gds-agent-package /path/to/gds_agent.whl` to benchmark a local wheel, or `--skill-file /path/to/SKILL.md` to use another version of the skill.

Citations example (GPT-5): custom dataset, so `--questions-file` is required. `--with-cypher-mcp` also starts the read-only Cypher MCP.

```bash
python benchmark_gds_agent.py --dataset citations \
  --questions-file questions/gds-questions-citations.json \
  --with-cypher-mcp --model gpt-5 \
  --gds-agent-package /path/to/gds_agent.whl \
  --skill-file /path/to/SKILL.md
```

```bash
python evaluate_benchmark.py citations \
  --questions-file questions/gds-questions-citations.json \
  --model gpt-5 --question-trace-report
```
