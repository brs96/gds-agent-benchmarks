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

    def send_question_to_openai_subprocess(self, config_file: Path, question: str) -> Optional[dict]:            
        try:
            return asyncio.run(self._send_question_to_openai_async(config_file, question))
        except Exception as e:
            logger.error(f"Error in OpenAI integration: {e}")
            return None

    async def _send_question_to_openai_async(self, config_file: Path, question: str) -> Optional[dict]:
        try:
            logger.debug(f"Starting OpenAI async request for: {question[:50]}")
            
            # Add timeout and environment variables
            async with MCPServerStdio(
                name="gds",
                params={
                    "command": "uvx",
                    "args": ["--from", "./gds_agent-0.4.0-py3-none-any.whl", "gds-agent"],
                    "env": {
                        "NEO4J_URI": "bolt://localhost:7687",
                        "NEO4J_USERNAME": "neo4j",
                        "NEO4J_PASSWORD": "12345678",
                    }
                },
            ) as server:
                
                formatted_prompt = f"You MUST use the available MCP tools to query the actual Neo4j database to answer this question. Do not rely on output from previous questions. Do not provide a hypothetical answer. Question: {question}"
                
                agent = Agent(
                    name="GPT-NEO4J-GDS",
                    instructions="You are a helpful assistant that uses the GDS (Graph Data Science) MCP tools to answer questions about Neo4j graph databases.",
                    mcp_servers=[server],
                    model=self.model,
                )
                
                logger.debug("Running agent with question...")
                result = await asyncio.wait_for(
                    Runner.run(agent, formatted_prompt), 
                    timeout=300
                )
                
                # Convert result to expected format
                response_data = {
                    "tool_calls": getattr(result, 'tool_calls', []) or [],
                    "tool_results": getattr(result, 'tool_results', []) or [],
                    "final_result": str(result.final_output) if hasattr(result, 'final_output') else str(result),
                    "num_turns": getattr(result, 'num_turns', 0) or 0,
                    "duration_ms": getattr(result, 'duration_ms', 0) or 0,
                    "raw_stream": str(result)
                }
                
                logger.debug(f"OpenAI request completed successfully")
                return response_data
                
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out after 300 seconds")
            return None
        except Exception as e:
            logger.error(f"Error in OpenAI async request: {e}")
            return None

    def send_question_to_provider(self, config_file: Path, question: str) -> Optional[dict]:
        """Route question to appropriate provider based on model."""
        if self.provider == 'openai':
            return self.send_question_to_openai_subprocess(config_file, question)
        else:  # claude or default
            return self.send_question_to_claude_subprocess(config_file, question)

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
        
        config_file = None
        
        try:
            config_file = self.create_mcp_config()
            
            results = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                response = self.send_question_to_provider(config_file, question)
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
            logger.error(f"Benchmark failed: {e}")
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
