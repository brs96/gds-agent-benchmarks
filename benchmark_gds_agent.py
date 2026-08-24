import json
import logging
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import os
import csv
import asyncio
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
import time


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Same package/version as gds-agent's recommended pairing (mcp.json / README).
DEFAULT_GDS_AGENT_PACKAGE = "gds-agent"
DEFAULT_SKILL_FILE = (
    Path(__file__).parent
    / "skills"
    / "neo4j-graph-data-scientist"
    / "SKILL.md"
)
CYPHER_MCP_PACKAGE = "mcp-neo4j-cypher@0.6.0"
CYPHER_TOOL_NAMES = {
    "read_neo4j_cypher",
    "write_neo4j_cypher",
    "get_neo4j_schema",
}


class GDSBenchmark:
    def __init__(self, 
                 dataset: str = "ln",
                 model: str = "sonnet-4-20250514",
                 results_file: str = None,
                 with_cypher_mcp: bool = False,
                 gds_agent_package: str = DEFAULT_GDS_AGENT_PACKAGE,
                 skill_file: Path = DEFAULT_SKILL_FILE):
        # Map dataset names to question files
        dataset_files = {
            "ln": "gds-algo-questions-ln.csv",
            "got": "gds-algo-questions-got.csv"
        }
        
        if dataset not in dataset_files:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_files.keys())}")
        
        self.dataset = dataset
        self.model = model
        self.provider = self._detect_provider(model)
        self.questions_file = Path(dataset_files[dataset])
        
        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = Path(f"results_{model}/{dataset}_results_{timestamp}.json")
        else:
            self.results_file = Path(results_file)
            
        self.with_cypher_mcp = with_cypher_mcp
        self.gds_agent_package = gds_agent_package
        self.skill_file = Path(skill_file)
        self.skill_instructions = self._load_skill()
        self.results = []

    def _detect_provider(self, model: str) -> str:
        """Detect the provider based on model name."""
        
        if any(model.startswith(prefix) for prefix in ['sonnet', 'haiku', 'opus']):
            return 'claude'
        
        if any(model.startswith(prefix) for prefix in ['gpt']):
            return 'openai'
        
        raise ValueError(f"Unknown model: {model}")

    def _load_skill(self) -> str:
        """Load the GDS agent skill used as instructions by every model."""
        if not self.skill_file.exists():
            raise ValueError(f"GDS skill file not found: {self.skill_file}")
        return self.skill_file.read_text(encoding="utf-8")

    def _mcp_environment(self) -> Dict[str, str]:
        """Build the MCP environment from environment variables."""
        required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        optional = ["NEO4J_DATABASE"]
        names = required + optional
        return {name: os.environ[name] for name in names if os.environ.get(name)}

    def _gds_mcp_params(self) -> dict:
        return {
            "command": "uvx",
            "args": ["--from", self.gds_agent_package, "gds-agent"],
            "env": self._mcp_environment(),
        }

    def _cypher_mcp_params(self) -> dict:
        """Stdio params for the read-only neo4j-cypher MCP server."""
        return {
            "command": "uvx",
            "args": [CYPHER_MCP_PACKAGE, "--transport", "stdio"],
            "env": {
                **self._mcp_environment(),
                "NEO4J_READ_ONLY": "true",
                "NEO4J_RESPONSE_TOKEN_LIMIT": "20000",
            },
        }


    async def _wait_for_server_ready(self, server, timeout: float = 60.0):
        """Wait for MCP server to be ready by testing actual functionality."""
        start_time = asyncio.get_event_loop().time()
        retry_delay = 2.0
        
        logger.debug(f"Polling for server readiness for up to {timeout} seconds...")
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # Test if we can list tools from the server
                logger.debug("Attempting to list tools...")
                tools = await asyncio.wait_for(server.list_tools(), timeout=10.0)
                if tools and len(tools) > 0:
                    logger.info(f"MCP server ready! Found {len(tools)} tools: {[t.name if hasattr(t, 'name') else str(t) for t in tools[:3]]}")
                    return
                else:
                    logger.debug("Server responded but no tools found yet")

            except Exception as e:
                logger.debug(f"Server readiness check failed: {str(e)[:100]}, retrying in {retry_delay}s...")
            
            await asyncio.sleep(retry_delay)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        raise asyncio.TimeoutError(f"MCP server not ready after {elapsed:.1f} seconds")

    def load_questions_from_file(self, questions_file: Path) -> List[str]:
        """Load questions from a single CSV file with 4-lines-per-question format."""
        logger.info(f"Loading questions from {questions_file}")
        questions = []
        
        if not questions_file.exists():
            logger.error(f"Questions file not found: {questions_file}")
            return []
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
                
                if not lines:
                    logger.error(f"Questions file is empty: {questions_file}")
                    return []

                # Process lines in groups of 4: question, tools, parameters, answer
                i = 1
                while i < len(lines):
                    if i + 3 < len(lines):  # Ensure we have all 4 lines
                        question = lines[i].strip()
                        if question:
                            questions.append(question)
                        i += 4
                    else:
                        # Handle incomplete block at end - check if it's a question line
                        if not lines[i].startswith('[') and not lines[i].startswith('{'):
                            # This might be a question (not tools or params)
                            question = lines[i].strip()
                            if question:
                                questions.append(question)
                        break
                        
        except Exception as e:
            logger.error(f"Error loading questions from {questions_file}: {e}")
            return []
            
        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        return questions

    def load_questions(self) -> List[str]:
        """Load questions from the configured CSV file."""
        return self.load_questions_from_file(self.questions_file)

    def create_mcp_config(self) -> Path:
        config = {
            "mcpServers": {
                "gds-agent": self._gds_mcp_params(),
            }
        }
        if self.with_cypher_mcp:
            config["mcpServers"]["neo4j-cypher"] = self._cypher_mcp_params()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            logger.debug(f"Created MCP config: {config}")
            return Path(f.name)


    def send_question_to_claude_subprocess(self, config_file: Path, question: str) -> Optional[dict]:
        try:
            logger.debug(f"Sending question to Claude via subprocess: {question}")
            
            formatted_prompt = f"You MUST use the available MCP tools to query the actual Neo4j database to answer this question. Do not rely on output from previous questions. Do not provide a hypothetical answer. Question: {question}"
            
            cmd = [
                "claude", f"--model", f"claude-{self.model}", "-p", "--verbose", "--output-format", "stream-json",
                "--mcp-config", str(config_file), 
                "--append-system-prompt", self.skill_instructions,
                "--dangerously-skip-permissions",
                "--allowedTools", "mcp__*"  # Allow all MCP tools
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            logger.debug(f"Input: {formatted_prompt}")
            logger.debug(f"Config file exists: {config_file.exists()}")
            logger.debug(f"Config file path: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    config_contents = f.read()
                    logger.debug(f"Config file contents: {config_contents}")
            except Exception as e:
                logger.error(f"Could not read config file: {e}")
            
            logger.info("Starting subprocess call...")
            result = subprocess.run(cmd, input=f"{formatted_prompt}\n", 
                                  capture_output=True, text=True, timeout=300)
            logger.info("Subprocess call completed")
            
            if result.stderr:
                logger.debug(f"Claude stderr: {result.stderr}")
            
            logger.debug(f"Return code: {result.returncode}")
            logger.debug(f"Stdout length: {len(result.stdout)}")
            
            if result.returncode == 0:
                logger.debug(f"Claude stream JSON response length: {len(result.stdout)} chars")
                
                return self.parse_stream_json_response(result.stdout)
            else:
                logger.debug(f"Claude error: {result.stderr}")
                return None
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Question timed out after 300s: {question[:50]}")
            logger.debug("This might indicate MCP server connection issues or very complex queries")
            return None
        except Exception as e:
            logger.error(f"Error communicating with Claude: {e}")
            return None

    def parse_stream_json_response(self, stream_output: str) -> dict:
        parsed_data = {
            "tool_calls": [],
            "tool_results": [],
            "final_result": "",
            "num_turns": 0,
            "duration_ms": 0,
            "raw_stream": stream_output
        }
        
        try:
            lines = stream_output.strip().split('\n')
            logger.debug(f"Parsing {len(lines)} lines from stream JSON")
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    # Extract tool calls from assistant messages
                    if data.get('type') == 'assistant' and 'message' in data:
                        content = data['message'].get('content', [])
                        for item in content:
                            if item.get('type') == 'tool_use':
                                tool_call = {
                                    "name": item['name'],
                                    "parameters": item.get('input', {}),
                                    "id": item['id']
                                }
                                parsed_data["tool_calls"].append(tool_call)
                                logger.debug(f"Found tool call: {item['name']} (line {line_num})")
                    # Extract tool results from user messages (these are tool responses)
                    elif data.get('type') == 'user' and 'message' in data:
                        content = data['message'].get('content', [])
                        for item in content:
                            if item.get('type') == 'tool_result':
                                # Handle both string and object content
                                content_data = item.get('content', [])
                                if content_data and isinstance(content_data, list) and len(content_data) > 0:
                                    result_text = content_data[0].get('text', '') if isinstance(content_data[0], dict) else str(content_data[0])
                                else:
                                    result_text = str(item.get('content', ''))
                                
                                tool_result = {
                                    "tool_use_id": item.get('tool_use_id', ''),
                                    "result": result_text
                                }
                                parsed_data["tool_results"].append(tool_result)
                                logger.debug(f"Found tool result for: {item.get('tool_use_id', 'unknown')} (line {line_num})")
                    
                    elif data.get('type') == 'result':
                        parsed_data["final_result"] = data.get('result', '')
                        parsed_data["num_turns"] = data.get('num_turns', 0)
                        parsed_data["duration_ms"] = data.get('duration_ms', 0)
                        
                except json.JSONDecodeError:
                    logger.error(f"Skipped non-JSON line: {line[:100] if line else 'None'}...")
                    
        except Exception as e:
            logger.error(f"Error parsing stream JSON: {e}")
            logger.debug(f"Failed parsing line: {line[:100] if line else 'None'}...")
            
        return parsed_data

    async def _send_question_to_openai_async(self, config_file: Path, question: str, gds_server: MCPServerStdio, cypher_server: Optional[MCPServerStdio] = None) -> Optional[dict]:
        try:
            logger.debug(f"Starting OpenAI async request for: {question[:50]}")

            formatted_prompt = f"You MUST use the available MCP tools to query the actual Neo4j database to answer this question. Do not rely on output from previous questions. Do not provide a hypothetical answer. Question: {question}"

            mcp_servers = [gds_server]
            if cypher_server is not None:
                mcp_servers.append(cypher_server)
            
            agent = Agent(
                name="GPT-NEO4J-GDS",
                instructions=(
                    "You are a helpful assistant that uses the GDS (Graph Data Science) "
                    "MCP tools to answer questions about Neo4j graph databases.\n\n"
                    f"{self.skill_instructions}"
                ),
                mcp_servers=mcp_servers,
                model=self.model,
            )
            
            logger.debug("Running agent with question...")
            result = await asyncio.wait_for(
                Runner.run(agent, formatted_prompt), 
                timeout=300
            )
            logger.debug(f"Result: {result}")
            response_data = {
                "tool_calls": [],
                "tool_results": [],
                "final_result": [],
                "num_turns": 0,
                "duration_ms": 0,
                "raw_stream": ""
            }
            raw_responses = result.raw_responses
            for raw_response in raw_responses:
                outputs = raw_response.output
                for output in outputs:
                    if output.type == "function_call":
                        try:
                            parsed_args = json.loads(output.arguments) if isinstance(output.arguments, str) else output.arguments
                        except (json.JSONDecodeError, TypeError):
                            parsed_args = output.arguments
                        
                        prefix = "mcp__neo4j-cypher__" if output.name in CYPHER_TOOL_NAMES else "mcp__gds-agent__"
                        response_data["tool_calls"].append({
                            "name": prefix + output.name,
                            "parameters": parsed_args,
                        })
                    elif output.type == "message":
                        response_data["final_result"] = output.content[0].text

            logger.debug(f"OpenAI request completed successfully")
            return response_data
                
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out after 300 seconds")
            return None
        except Exception as e:
            logger.error(f"The exception type is: {type(e)}")
            logger.error(f"Error in OpenAI async request: {e}")
            return None


    def create_result_record(self, question: str, response: dict) -> Dict[str, Any]:
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "response_data": response,
            "success": response is not None
        }


    def run_benchmark(self) -> List[Dict[str, Any]]:
        logger.info(f"Starting GDS Agent benchmark for {self.dataset} dataset...")
        
        questions = self.load_questions()
        if not questions:
            logger.error("No questions to process")
            return []
        
        if self.provider == 'openai':
            return asyncio.run(self._run_benchmark_openai(questions))
        else:
            return self._run_benchmark_claude(questions)

    async def _run_benchmark_openai(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run benchmark for OpenAI models with fresh MCP server for each question."""
        results = []

        for i, question in enumerate(questions, 1):
            time.sleep(2)
            logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            logger.debug(f"Starting fresh MCP server for question {i}")

            try:
                async with AsyncExitStack() as stack:
                    gds_server = await stack.enter_async_context(
                        MCPServerStdio(name="gds", params=self._gds_mcp_params(), client_session_timeout_seconds=600)
                    )
                    await self._wait_for_server_ready(gds_server, timeout=60.0)
                    logger.info("GDS MCP server started and ready")

                    cypher_server = None
                    if self.with_cypher_mcp:
                        cypher_server = await stack.enter_async_context(
                            MCPServerStdio(name="neo4j-cypher", params=self._cypher_mcp_params(), client_session_timeout_seconds=600)
                        )
                        await self._wait_for_server_ready(cypher_server, timeout=60.0)
                        logger.info("Cypher MCP server started and ready")

                    response = await self._send_question_to_openai_async(None, question, gds_server, cypher_server)

                result = self.create_result_record(question, response)
                result["dataset"] = self.dataset
                result["model"] = self.model
                result["provider"] = self.provider
                result["with_cypher_mcp"] = self.with_cypher_mcp
                results.append(result)

                if response:
                    num_tools = len(response.get('tool_calls', []))
                    num_turns = response.get('num_turns', 0)
                    logger.info(f"Question {i}: ✓ ({num_tools} tools, {num_turns} turns)")
                else:
                    logger.info(f"Question {i}: ✗ (no response)")

            except Exception as e:
                logger.error(f"Question {i} failed: {e}")
                result = self.create_result_record(question, None)
                result["dataset"] = self.dataset
                result["model"] = self.model
                result["provider"] = self.provider
                result["with_cypher_mcp"] = self.with_cypher_mcp
                results.append(result)
                logger.info(f"Question {i}: ✗ (exception: {str(e)[:50]})")

        self.results = results
        return results

    def _run_benchmark_claude(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run benchmark for Claude models using subprocess approach."""
        config_file = None
        
        try:
            config_file = self.create_mcp_config()
            
            results = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                response = self.send_question_to_claude_subprocess(config_file, question)
                result = self.create_result_record(question, response)
                result["dataset"] = self.dataset
                result["model"] = self.model
                result["provider"] = self.provider
                result["with_cypher_mcp"] = self.with_cypher_mcp
                results.append(result)
                
                if response:
                    num_tools = len(response.get('tool_calls', []))
                    num_turns = response.get('num_turns', 0)
                    logger.info(f"Question {i}: ✓ ({num_tools} tools, {num_turns} turns)")
                else:
                    logger.info(f"Question {i}: ✗ (no response)")
                
                import time
                time.sleep(1)
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Subprocess benchmark failed: {e}")
            return []
            
        finally:
            if config_file and config_file.exists():
                try:
                    os.unlink(config_file)
                except:
                    pass


    def save_results(self) -> None:
        if not self.results:
            logger.warning("No results to save")
            return
        
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {self.results_file}")
        
        total_questions = len(self.results)
        successful_responses = sum(1 for r in self.results if r["success"])
        
        summary = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset,
                "model": self.model,
                "provider": self.provider,
                "questions_file": str(self.questions_file),
                "with_cypher_mcp": self.with_cypher_mcp,
                "skill_file": str(self.skill_file),
                "total_questions": total_questions,
                "successful_responses": successful_responses,
                "response_rate": successful_responses / total_questions if total_questions > 0 else 0,
            },
            "raw_results": self.results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved. Response rate: {successful_responses}/{total_questions} "
                   f"({summary['benchmark_info']['response_rate']:.2%})")

    def print_summary(self) -> None:
        """Print a summary of the benchmark results."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        total = len(self.results)
        
        print("\n" + "="*60)
        print("GDS AGENT BENCHMARK SUMMARY")
        print("="*60)
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model}")
        print(f"Cypher MCP: {'on' if self.with_cypher_mcp else 'off'}")
        print(f"Total Questions: {total}")
        print("\nDetailed Results:")
        
        for i, result in enumerate(self.results, 1):
            status = "✓" if result["success"] else "✗"
            question = result['question']
            print(f"{i:2d}. {status} {question}")
            
            if result["success"] and result["response_data"]:
                data = result["response_data"]
                num_tools = len(data.get('tool_calls', []))
                num_turns = data.get('num_turns', 0)
                duration = data.get('duration_ms', 0)
                print(f"    → Tools: {num_tools}, Turns: {num_turns}, Duration: {duration}ms")
                
                # Show tool names if any
                if data.get('tool_calls'):
                    tool_names = [call['name'] for call in data['tool_calls']]
                    print(f"    → Called: {', '.join(tool_names)}")
            else:
                print(f"    → No response received")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="GDS Agent Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples: 
        python benchmark_gds_agent.py ln --model gpt-4o                  # Run LN questions with OpenAI GPT-4o
        python benchmark_gds_agent.py ln --model gpt-4o --with-cypher-mcp  # Also enable read-only Cypher MCP"""
    )
    
    parser.add_argument(
        "dataset",
        choices=["ln", "got"],
        help="Dataset to run: 'ln' for London network questions, 'got' for Game of Thrones questions"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="sonnet-4-20250514",
        help="Model to use (default: sonnet-4-20250514). Examples: sonnet-4-20250514, gpt-4o"
    )

    parser.add_argument(
        "--with-cypher-mcp",
        action="store_true",
        help="Also start the read-only neo4j-cypher MCP server alongside gds-agent"
    )

    parser.add_argument(
        "--gds-agent-package",
        default=DEFAULT_GDS_AGENT_PACKAGE,
        help="Package passed to uvx --from (default: gds-agent). May also be a path to a local checkout or wheel."
    )

    parser.add_argument(
        "--skill-file",
        type=Path,
        default=DEFAULT_SKILL_FILE,
        help="Path to the neo4j-graph-data-scientist SKILL.md file"
    )
    
    args = parser.parse_args()
    load_dotenv()

    print("GDS Agent Benchmarking Tool")
    print("="*40)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Cypher MCP: {'on' if args.with_cypher_mcp else 'off'}")
    
    try:
        benchmark = GDSBenchmark(
            dataset=args.dataset,
            model=args.model,
            with_cypher_mcp=args.with_cypher_mcp,
            gds_agent_package=args.gds_agent_package,
            skill_file=args.skill_file,
        )
        print(f"Provider: {benchmark.provider.upper()}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Check if questions file exists
    if not benchmark.questions_file.exists():
        print(f"❌ Questions file not found: {benchmark.questions_file}")
        sys.exit(1)
    
    print(f"GDS agent: {args.gds_agent_package}")
    print(f"Skill file: {benchmark.skill_file}")
    print(f"Questions file: {benchmark.questions_file}")
    print(f"Results file: {benchmark.results_file}")
    print("\nStarting benchmark...")
    
    try:
        benchmark.run_benchmark()
        benchmark.save_results()
        benchmark.print_summary()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
