#!/usr/bin/env python3
"""
GDS Agent Benchmarking Tool

This script benchmarks Claude's performance with the GDS (Graph Data Science) MCP server.
Usage:
1. Start the GDS server: ./start_server.sh
2. Run this benchmark: python benchmark_gds_agent.py
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import os
import csv

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Enable debug logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GDSBenchmark:
    def __init__(self, 
                 questions_file: str = "gds-algo-questions-basic.csv",
                 results_file: str = "benchmark_results.json"):
        self.questions_file = Path(questions_file)
        self.results_file = Path(results_file)
        self.results = []
        
        # Expected tool mappings for different question types
        self.expected_tools = {
            "shortest path": ["find_shortest_path", "dijkstra_single_source_shortest_path"],
            "path": ["find_shortest_path", "dijkstra_single_source_shortest_path", "breadth_first_search"],
            "route": ["find_shortest_path", "dijkstra_single_source_shortest_path"],
            "distance": ["find_shortest_path", "dijkstra_single_source_shortest_path"],
            "connection": ["breadth_first_search", "depth_first_search", "find_shortest_path"],
            "reachable": ["breadth_first_search", "depth_first_search"],
            "spanning": ["minimum_weight_spanning_tree"],
            "centrality": ["pagerank", "betweenness_centrality", "closeness_centrality"],
            "community": ["louvain", "label_propagation"],
        }

    def load_questions(self) -> List[str]:
        """Load questions from CSV file."""
        logger.info(f"Loading questions from {self.questions_file}")
        questions = []
        
        if not self.questions_file.exists():
            logger.error(f"Questions file not found: {self.questions_file}")
            return []
        
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as file:
                # Try to parse as CSV first
                try:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if 'question' in row and row['question'].strip():
                            questions.append(row['question'].strip())
                except csv.Error:
                    # Fallback to old plain text format
                    file.seek(0)
                    content = file.read().strip()
                    if not content:
                        logger.error("Questions file is empty")
                        return []
                    
                    # Handle different formats
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Skip header-like lines
                    for line in lines:
                        if not line.lower().startswith(('question', 'query', '#')):
                            questions.append(line)
                        
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []
            
        logger.info(f"Loaded {len(questions)} questions")
        return questions

    def create_mcp_config(self) -> Path:
        """Create MCP configuration for connecting to running server."""
        config = {
            "mcpServers": {
                "gds-agent": {
                    "command": "uvx",
                    "args": ["--from", "./gds_agent-0.2.0-py3-none-any.whl", "gds-agent"],
                    "env": {
                        "NEO4J_URI": "bolt://localhost:7687",
                        "NEO4J_USERNAME": "neo4j", 
                        "NEO4J_PASSWORD": "12345678",
                    }
                }
            }
        }
        
        # Use temporary file for config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            logger.debug(f"Created MCP config: {config}")
            return Path(f.name)


    def send_question_to_claude_subprocess(self, config_file: Path, question: str) -> Optional[dict]:
        """Send question using stream-json mode to capture detailed tool calls."""
        try:
            logger.debug(f"Sending question via subprocess: {question}")
            
            # Frame question to strongly encourage tool usage
            enhanced_question = f"You MUST use the available MCP tools to query the actual Neo4j database to answer this question. Do not provide a hypothetical answer. Question: {question}"
            
            # Use stream-json mode to capture all tool calls and intermediate steps
            cmd = [
                "claude", "-p", "--verbose", "--output-format", "stream-json",
                "--mcp-config", str(config_file), 
                "--dangerously-skip-permissions",
                "--allowedTools", "mcp__*"  # Allow all MCP tools
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            logger.debug(f"Input: {enhanced_question}")
            logger.debug(f"Config file exists: {config_file.exists()}")
            logger.debug(f"Config file path: {config_file}")
            
            # Read and log config file contents
            try:
                with open(config_file, 'r') as f:
                    config_contents = f.read()
                    logger.debug(f"Config file contents: {config_contents}")
            except Exception as e:
                logger.error(f"Could not read config file: {e}")
            
            logger.info("Starting subprocess call...")
            result = subprocess.run(cmd, input=f"{enhanced_question}\n", 
                                  capture_output=True, text=True, timeout=120)
            logger.info("Subprocess call completed")
            
            # Log stderr to see any MCP connection issues
            if result.stderr:
                logger.debug(f"Claude stderr: {result.stderr}")
            
            logger.debug(f"Return code: {result.returncode}")
            logger.debug(f"Stdout length: {len(result.stdout)}")
            
            if result.returncode == 0:
                logger.debug(f"Claude stream JSON response length: {len(result.stdout)} chars")
                
                # Parse the stream JSON to extract tool calls and final result
                return self.parse_stream_json_response(result.stdout)
            else:
                logger.debug(f"Claude error: {result.stderr}")
                return None
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Question timed out after 120s: {question[:50]}")
            logger.debug("This might indicate MCP server connection issues or very complex queries")
            return None
        except Exception as e:
            logger.error(f"Error communicating with Claude: {e}")
            return None

    def parse_stream_json_response(self, stream_output: str) -> dict:
        """Parse stream JSON output to extract tool calls, results, and final answer."""
        parsed_data = {
            "tool_calls": [],
            "tool_results": [],
            "final_result": "",
            "num_turns": 0,
            "duration_ms": 0,
            "raw_stream": stream_output,
            "available_tools": []
        }
        
        try:
            lines = stream_output.strip().split('\n')
            logger.debug(f"Parsing {len(lines)} lines from stream JSON")
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract available tools from init message
                    if data.get('type') == 'system' and data.get('subtype') == 'init':
                        parsed_data["available_tools"] = data.get('tools', [])
                    
                    # Extract tool calls from assistant messages
                    elif data.get('type') == 'assistant' and 'message' in data:
                        content = data['message'].get('content', [])
                        for item in content:
                            if item.get('type') == 'tool_use':
                                tool_call = {
                                    "name": item['name'],
                                    "parameters": item.get('input', {}),
                                    "id": item['id']
                                }
                                # Only add if not already present (avoid duplicates)
                                if tool_call not in parsed_data["tool_calls"]:
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
                                # Only add if not already present (avoid duplicates)
                                if tool_result not in parsed_data["tool_results"]:
                                    parsed_data["tool_results"].append(tool_result)
                                    logger.debug(f"Found tool result for: {item.get('tool_use_id', 'unknown')} (line {line_num})")
                    
                    # Extract final result and metadata
                    elif data.get('type') == 'result':
                        parsed_data["final_result"] = data.get('result', '')
                        parsed_data["num_turns"] = data.get('num_turns', 0)
                        parsed_data["duration_ms"] = data.get('duration_ms', 0)
                        
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing stream JSON: {e}")
            logger.debug(f"Failed parsing line: {line[:100] if line else 'None'}...")
            
        return parsed_data

    def send_question_to_claude(self, config_file: Path, question: str) -> Optional[dict]:
        """Send a question to Claude - uses stream-json mode to capture detailed tool calls."""
        # Always use subprocess with stream-json mode for detailed tool capture
        return self.send_question_to_claude_subprocess(config_file, question)

    def create_result_record(self, question: str, response: dict) -> Dict[str, Any]:
        """Create a simple result record with raw response data."""
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "response_data": response,  # Store the complete parsed response
            "success": response is not None
        }

    def run_benchmark(self) -> List[Dict[str, Any]]:
        """Run the complete benchmark."""
        logger.info("Starting GDS Agent benchmark...")
        
        # Load questions
        questions = self.load_questions()
        if not questions:
            logger.error("No questions to process")
            return []
        
        # Create MCP config once for all questions
        config_file = None
        
        try:
            config_file = self.create_mcp_config()
            
            results = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                response = self.send_question_to_claude(config_file, question)
                result = self.create_result_record(question, response)
                results.append(result)
                
                # Log basic status
                if response:
                    num_tools = len(response.get('tool_calls', []))
                    num_turns = response.get('num_turns', 0)
                    logger.info(f"Question {i}: ✓ ({num_tools} tools, {num_turns} turns)")
                else:
                    logger.info(f"Question {i}: ✗ (no response)")
                
                # Small delay between questions
                import time
                time.sleep(1)
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return []
            
        finally:
            # Cleanup config file
            if config_file and config_file.exists():
                try:
                    os.unlink(config_file)
                except:
                    pass

    def save_results(self) -> None:
        """Save benchmark results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving results to {self.results_file}")
        
        total_questions = len(self.results)
        successful_responses = sum(1 for r in self.results if r["success"])
        
        summary = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
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
    """Main entry point."""
    print("GDS Agent Benchmarking Tool")
    print("="*40)
    
    # Check if questions file exists
    if not Path("gds-algo-questions-basic.csv").exists():
        print("❌ Questions file 'gds-algo-questions-basic.csv' not found")
        print("Please create a file with your test questions.")
        sys.exit(1)
    
    # Check if wheel file exists
    if not Path("gds_agent-0.2.0-py3-none-any.whl").exists():
        print("❌ GDS agent wheel file not found")
        print("Please ensure 'gds_agent-0.2.0-py3-none-any.whl' is in the current directory")
        sys.exit(1)
    
    print("✅ Setup looks good!")
    print("\n📋 Starting benchmark...")
    
    benchmark = GDSBenchmark()
    
    try:
        benchmark.run_benchmark()
        benchmark.save_results()
        benchmark.print_summary()
        
    except KeyboardInterrupt:
        print("\n⛔ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
