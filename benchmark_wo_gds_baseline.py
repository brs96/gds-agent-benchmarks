import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from agents import Agent, MaxTurnsExceeded, ModelSettings, Runner, RunHooks, function_tool
from dotenv import load_dotenv
from openai.types.shared.reasoning import Reasoning

from neo4j import GraphDatabase
from benchmark_gds_agent import DEFAULT_MAX_TURNS, GDSBenchmark, TokenTracker, logger  # logs to benchmark_results.log


def terminate_running_queries() -> None:
    """A timeout only stops the waiting: stop the queries still running on the database."""
    try:
        with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])) as driver, driver.session() as session:
            ids = [r["transactionId"] for r in session.run(
                "SHOW TRANSACTIONS YIELD transactionId, currentQuery "
                "WHERE NOT currentQuery STARTS WITH 'SHOW TRANSACTIONS' RETURN transactionId"
            )]
            if ids:
                session.run("TERMINATE TRANSACTIONS $ids", ids=ids).consume()
                logger.info(f"Terminated {len(ids)} running queries")
    except Exception as e:
        logger.warning(f"Could not terminate running queries: {e}")

AGENT_PYTHON = Path(__file__).parent / ".venv-baseline" / "bin" / "python"

INSTRUCTIONS = """You are a coding agent answering questions about a Neo4j graph database.
Your only tool is Python. Installed packages: neo4j, graphdatascience, pandas.
Credentials are in the environment variables NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (and NEO4J_DATABASE if set)."""

PROMPT = "You MUST use Python to query the actual Neo4j database and run GDS algorithms to answer this question. Do not rely on output from previous questions. Do not provide a hypothetical answer. Question: {}"

DROP_PROJECTIONS = """
import os
from neo4j import GraphDatabase
with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])) as driver:
    driver.execute_query(
        "CALL gds.graph.list() YIELD graphName CALL gds.graph.drop(graphName) YIELD graphName AS dropped RETURN dropped",
        database_=os.environ.get("NEO4J_DATABASE"),
    )
"""


def neo4j_env():
    """Only the Neo4j credentials, so agent code can't see the LLM API keys."""
    names = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE", "PATH", "HOME"]
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def drop_all_projections():
    subprocess.run([AGENT_PYTHON, "-c", DROP_PROJECTIONS], env=neo4j_env(), check=True)


python_calls = 0 


@function_tool
async def run_python(code: str) -> str:
    """Run a Python script and return its stdout and stderr."""
    global python_calls
    python_calls += 1
    n = python_calls
    logger.info(f"[baseline]   python call #{n} started ({len(code.splitlines())} lines)")
    started = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        AGENT_PYTHON, "-c", code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=neo4j_env()
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
        out = out.decode()
        error = " with error" if "Traceback" in out or "Error" in out else ""
        logger.info(f"[baseline]   python call #{n} done{error} in {time.perf_counter() - started:.0f}s ({len(out)} chars output)")
        return out[-20000:]
    except asyncio.TimeoutError:
        logger.warning(f"[baseline]   python call #{n} timed out after 600s")
        return "Error: timed out after 600s"
    finally:
        if proc.returncode is None:  # timed out or question cancelled
            proc.kill()
            terminate_running_queries()  # the killed script's queries keep running on the database


def ask_openai(model, question, max_turns, reasoning_effort=None):
    settings = ModelSettings(reasoning=Reasoning(effort=reasoning_effort)) if reasoning_effort else ModelSettings()
    agent = Agent(name="python-baseline", instructions=INSTRUCTIONS, tools=[run_python], model=model, model_settings=settings)
    started = time.perf_counter()
    tracker = TokenTracker()
    try:
        result = asyncio.run(asyncio.wait_for(
            Runner.run(agent, PROMPT.format(question), max_turns=max_turns, hooks=tracker),
            timeout=1500,
        ))
        data = GDSBenchmark._openai_run_to_response_data(None, result, int((time.perf_counter() - started) * 1000))
    except asyncio.TimeoutError:
        logger.warning("[baseline]   question timed out after 1500s")
        return tracker.timeout_response(int((time.perf_counter() - started) * 1000))
    except MaxTurnsExceeded as e:
        data = GDSBenchmark._openai_run_to_response_data(None, e.run_data, int((time.perf_counter() - started) * 1000))
        data["max_turns_exceeded"] = True
    for call in data["tool_calls"]:
        call["name"] = "run_python"
    return data


def ask_claude(model, question, max_turns, reasoning_effort=None):  # reasoning_effort: OpenAI only
    cmd = [
        "claude", "--model", f"claude-{model}", "-p", "--verbose", "--output-format", "stream-json",
        "--strict-mcp-config",
        "--tools", "Bash",
        "--allowedTools", "Bash(python:*)", "Bash(python3:*)",
        "--append-system-prompt", INSTRUCTIONS,
        "--max-turns", str(max_turns),
    ]
    env = {**os.environ, "PATH": f"{AGENT_PYTHON.parent}:{os.environ['PATH']}"}  # python = .venv-baseline
    r = subprocess.run(cmd, input=PROMPT.format(question), capture_output=True, text=True, timeout=1500, env=env)
    return GDSBenchmark.parse_stream_json_response(None, r.stdout)


def used_gds(data):
    code = json.dumps([c["parameters"] for c in data["tool_calls"]])
    return "gds." in code or "graphdatascience" in code


def tool_errors(data):
    return sum("Error" in r["result"] or "Traceback" in r["result"] for r in data["tool_results"])


def main():
    global python_calls
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    args = parser.parse_args()
    load_dotenv()

    ask = ask_claude if args.model.startswith(("sonnet", "haiku", "opus")) else ask_openai
    questions = GDSBenchmark.load_questions_from_file(None, args.questions_file)
    results = []
    total = args.runs * len(questions)
    run_started = time.perf_counter()
    logger.info(f"[baseline] Starting: model={args.model} dataset={args.dataset} questions={args.questions_file} "
                f"runs={args.runs} max_turns={args.max_turns} reasoning_effort={args.reasoning_effort or 'default'} "
                f"({total} questions in total)")


    for run in range(1, args.runs + 1):
        for i, question in enumerate(questions, 1):
            python_calls = 0
            elapsed = (time.perf_counter() - run_started) / 60
            logger.info(f"[baseline] Run {run}/{args.runs}, question {i}/{len(questions)} "
                        f"({len(results) + 1}/{total} overall, {elapsed:.1f} min elapsed): {question[:60]}")
            q_started = time.perf_counter()
            drop_all_projections()
            try:
                data = ask(args.model, question, args.max_turns, args.reasoning_effort)
            except Exception as e:
                logger.error(f"[baseline]   failed: {e}")
                data = None
            if data:
                data["used_gds"] = used_gds(data)
                data["tool_errors"] = tool_errors(data)
            success = bool(data and data["final_result"]) and not data.get("max_turns_exceeded", False)
            results.append({
                "question": question,
                "run": run,
                "timestamp": datetime.now().isoformat(),
                "response_data": data,
                "success": success,
                "model": args.model,
            })
            took = (time.perf_counter() - q_started) / 60
            if data:
                tokens = data["token_usage"].get("total_tokens", 0)
                logger.info(f"[baseline]   {'✓' if success else '✗'} done in {took:.1f} min: {len(data['tool_calls'])} python calls, "
                            f"{data['tool_errors']} errors, used_gds={data['used_gds']}, {data['num_turns']} turns, {tokens} tokens"
                            f"{', max turns exceeded' if data.get('max_turns_exceeded') else ''}"
                            f"{', timed out' if data.get('timed_out') else ''}")
            else:
                logger.info(f"[baseline]   ✗ no response after {took:.1f} min")

    out = Path(f"results_{args.model}/{args.dataset}-python_results_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "arm": "python",
        "questions_file": str(args.questions_file),
        "runs": args.runs,
        "max_turns": args.max_turns,
        "reasoning_effort": args.reasoning_effort or "default",
        "total_questions": len(results),
        "successful_responses": sum(r["success"] for r in results),
    }
    out.write_text(json.dumps({"benchmark_info": info, "raw_results": results}, indent=2))
    logger.info(f"[baseline] Finished in {(time.perf_counter() - run_started) / 60:.1f} min: "
                f"{info['successful_responses']}/{len(results)} successful. Saved {out}")


if __name__ == "__main__":
    main()
