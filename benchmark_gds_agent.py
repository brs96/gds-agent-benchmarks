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
from agents import Agent, Runner
from agents.mcp import MCPServerStdio


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GDSBenchmark:
    def __init__(self, 
                 dataset: str = "ln",
                 model: str = "sonnet-4-20250514",
                 results_file: str = None):
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
            
        self.results = []
        self.mcp_server = None  # For reusable MCP server

    def _detect_provider(self, model: str) -> str:
        """Detect the provider based on model name."""
        # OpenAI models
        openai_models = [
            'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',
            'o1-preview', 'o1-mini'
        ]
        
        # Claude models (existing patterns)
        if any(model.startswith(prefix) for prefix in ['sonnet', 'haiku', 'opus']):
            return 'claude'
        
        # Check for OpenAI models
        if model in openai_models:
            return 'openai'
        
        # Default to claude for backward compatibility
        return 'claude'

    async def start_mcp_server(self):
        """Start MCP server for GPT models to be reused across questions."""
        if self.provider == 'openai' and self.mcp_server is None:
            logger.info("Starting reusable MCP server for GPT models...")
            try:
                # Check if wheel file exists
                wheel_path = "./gds_agent-0.4.0-py3-none-any.whl"
                if not Path(wheel_path).exists():
                    raise Exception(f"Wheel file not found: {wheel_path}")
                
                logger.debug("Creating MCP server instance...")
                self.mcp_server = MCPServerStdio(
                    name="gds",
                    params={
                        "command": "uvx",
                        "args": ["--isolated", wheel_path],
                        "env": {
                            "NEO4J_URI": "bolt://localhost:7687",
                            "NEO4J_USERNAME": "neo4j",
                            "NEO4J_PASSWORD": "12345678",
                        }
                    },
                )
                logger.debug("Starting MCP server connection...")
                # Start the server process
                await self.mcp_server.__aenter__()
                
                # Wait for server to be ready with proper polling
                logger.debug("Waiting for MCP server to be ready...")
                await self._wait_for_server_ready(timeout=60.0)
                logger.info("MCP server started and ready")
            except asyncio.TimeoutError:
                logger.error("MCP server initialization timed out after 30 seconds")
                self.mcp_server = None
                raise Exception("Failed to initialize MCP server: timeout")
            except Exception as e:
                logger.error(f"Error initializing MCP server: {e}")
                self.mcp_server = None
                raise

    async def stop_mcp_server(self):
        """Stop MCP server for GPT models."""
        if self.mcp_server is not None:
            logger.info("Stopping MCP server...")
            try:
                await self.mcp_server.__aexit__(None, None, None)
                self.mcp_server = None
                logger.info("MCP server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping MCP server: {e}")
                self.mcp_server = None

    async def _wait_for_server_ready(self, timeout: float = 60.0):
        """Wait for MCP server to be ready by testing actual functionality."""
        start_time = asyncio.get_event_loop().time()
        retry_delay = 2.0
        max_retry_delay = 8.0
        
        logger.debug(f"Polling for server readiness for up to {timeout} seconds...")
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # Test if we can list tools from the server
                if hasattr(self.mcp_server, 'list_tools'):
                    logger.debug("Attempting to list tools...")
                    tools = await asyncio.wait_for(self.mcp_server.list_tools(), timeout=10.0)
                    if tools and len(tools) > 0:
                        logger.info(f"MCP server ready! Found {len(tools)} tools: {[t.name if hasattr(t, 'name') else str(t) for t in tools[:3]]}")
                        return
                    else:
                        logger.debug("Server responded but no tools found yet")
                
                # Alternative: try to call a simple tool if list_tools doesn't work
                elif hasattr(self.mcp_server, 'call_tool'):
                    logger.debug("Testing server with ping...")
                    # Try a simple operation that should work
                    try:
                        await asyncio.wait_for(self.mcp_server.call_tool("ping", {}), timeout=5.0)
                        logger.info("MCP server ready! Ping successful")
                        return
                    except Exception:
                        logger.debug("Server not ready for tool calls yet")
                        
            except asyncio.TimeoutError:
                logger.debug(f"Server readiness check timed out, retrying in {retry_delay}s...")
            except Exception as e:
                logger.debug(f"Server readiness check failed: {str(e)[:100]}, retrying in {retry_delay}s...")
            
            # Wait before retrying with exponential backoff
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.4, max_retry_delay)
        
        # If we get here, we timed out
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
                "gds-agent": {
                    "command": "uvx",
                    "args": ["--from", "./gds_agent-0.4.0-py3-none-any.whl", "gds-agent"],
                    "env": {
                        "NEO4J_URI": "bolt://localhost:7687",
                        "NEO4J_USERNAME": "neo4j", 
                        "NEO4J_PASSWORD": "12345678",
                    }
                }
            }
        }
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

    async def _send_question_to_openai_async(self, config_file: Path, question: str) -> Optional[dict]:
        try:
            logger.debug(f"Starting OpenAI async request for: {question[:50]}")
            
            # Ensure we have a running MCP server
            if self.mcp_server is None:
                raise Exception("MCP server not initialized. Call start_mcp_server() first.")
            
            formatted_prompt = f"You MUST use the available MCP tools to query the actual Neo4j database to answer this question. Do not rely on output from previous questions. Do not provide a hypothetical answer. Question: {question}"
            
            agent = Agent(
                name="GPT-NEO4J-GDS",
                instructions="You are a helpful assistant that uses the GDS (Graph Data Science) MCP tools to answer questions about Neo4j graph databases.",
                mcp_servers=[self.mcp_server],
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
                        # Parse arguments from JSON string to dictionary
                        try:
                            parsed_args = json.loads(output.arguments) if isinstance(output.arguments, str) else output.arguments
                        except (json.JSONDecodeError, TypeError):
                            parsed_args = output.arguments
                        
                        response_data["tool_calls"].append({
                            "name": output.name,
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
        
        # For OpenAI models, use async approach with persistent MCP server
        if self.provider == 'openai':
            return asyncio.run(self._run_benchmark_async(questions))
        else:
            # For Claude models, use the existing subprocess approach
            return self._run_benchmark_subprocess(questions)

    async def _run_benchmark_async(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run benchmark for OpenAI models with persistent MCP server."""
        try:
            # Start MCP server once at the beginning
            await self.start_mcp_server()
            
            results = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                response = await self._send_question_to_openai_async(None, question)
                result = self.create_result_record(question, response)
                result["dataset"] = self.dataset
                result["model"] = self.model
                result["provider"] = self.provider
                results.append(result)
                
                if response:
                    num_tools = len(response.get('tool_calls', []))
                    num_turns = response.get('num_turns', 0)
                    logger.info(f"Question {i}: ✓ ({num_tools} tools, {num_turns} turns)")
                else:
                    logger.info(f"Question {i}: ✗ (no response)")
                
                await asyncio.sleep(1)  # Small delay between questions
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Async benchmark failed: {e}")
            return []
            
        finally:
            # Clean up MCP server
            await self.stop_mcp_server()

    def _run_benchmark_subprocess(self, questions: List[str]) -> List[Dict[str, Any]]:
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
        python benchmark_gds_agent.py ln                            # Run LN questions with default Claude model
        python benchmark_gds_agent.py got                           # Run GoT questions with default Claude model
        python benchmark_gds_agent.py ln --model haiku-3-20241022   # Run LN questions with Claude Haiku
        python benchmark_gds_agent.py ln --model gpt-4o             # Run LN questions with OpenAI GPT-4o
        python benchmark_gds_agent.py got --model gpt-4-turbo       # Run GoT questions with OpenAI GPT-4 Turbo"""
    )
    
    parser.add_argument(
        "dataset",
        choices=["ln", "got"],
        help="Dataset to run: 'ln' for London network questions, 'got' for Game of Thrones questions"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="sonnet-4-20250514",
        help="Model to use (default: sonnet-4-20250514). Claude examples: sonnet-4-20250514, haiku-3-20241022. OpenAI examples: gpt-4o, gpt-4-turbo, gpt-3.5-turbo"
    )
    
    args = parser.parse_args()
    
    print("GDS Agent Benchmarking Tool")
    print("="*40)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    try:
        benchmark = GDSBenchmark(dataset=args.dataset, model=args.model)
        print(f"Provider: {benchmark.provider.upper()}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Check if wheel file exists
    if not Path("gds_agent-0.4.0-py3-none-any.whl").exists():
        print("❌ GDS agent wheel file not found")
        print("Please ensure 'gds_agent-0.4.0-py3-none-any.whl' is in the current directory")
        sys.exit(1)
    
    # Check if questions file exists
    if not benchmark.questions_file.exists():
        print(f"❌ Questions file not found: {benchmark.questions_file}")
        sys.exit(1)
    
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
