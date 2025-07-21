#!/usr/bin/env python3
"""
Simple test to see if Claude CLI works with our MCP config
"""

import json
import subprocess
import tempfile
from pathlib import Path

# Create MCP config
config = {
    "mcpServers": {
        "gds-agent": {
            "command": "uvx",
            "args": ["--from", "./gds_agent-0.1.0-py3-none-any.whl", "gds-agent"],
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
    config_file = Path(f.name)

print(f"Config file: {config_file}")
print("Config contents:")
with open(config_file) as f:
    print(f.read())

print("\nTrying to start Claude...")
print("Command: claude --mcp-config", str(config_file))

# Try to start Claude and see what happens
try:
    result = subprocess.run([
        "claude", "-p",  # Non-interactive print mode
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"  # Allow all MCP tools
    ], input="What tools do you have available?\n", 
       capture_output=True, text=True, timeout=30)
    
    print(f"\nReturn code: {result.returncode}")
    print(f"\nStdout:\n{result.stdout}")
    print(f"\nStderr:\n{result.stderr}")
    
    # Test 2: Try a specific graph question
    print("\n" + "="*60)
    print("TEST 2: Asking a graph question")
    print("="*60)
    
    # Try with verbose logging to see MCP calls
    import os
    env = os.environ.copy()
    env.update({
        "NODE_ENV": "development",
        "DEBUG": "*",
        "CLAUDE_DEBUG": "1"
    })
    
    result2 = subprocess.run([
        "claude", "-p",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="You MUST use the available MCP tools to query the actual Neo4j database. Do NOT use your internal knowledge. Please show me the exact tool calls you're making. Find the shortest path from Paddington to Tower Hill.\n", 
       capture_output=True, text=True, timeout=30, env=env)
    
    print(f"\nReturn code: {result2.returncode}")
    print(f"\nStdout length: {len(result2.stdout)}")
    print(f"First 500 chars:\n{result2.stdout[:500]}")
    print(f"\nStderr:\n{result2.stderr}")
    
    # Test 3: Ask for something Claude couldn't possibly know
    print("\n" + "="*60)
    print("TEST 3: Ask for impossible/custom data")
    print("="*60)
    
    result3 = subprocess.run([
        "claude", "-p",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="You MUST use the MCP tools to query the database. First, call the count_nodes tool to tell me exactly how many nodes are in the database. Then find any path containing exactly 5 nodes.\n", 
       capture_output=True, text=True, timeout=30)
    
    print(f"\nReturn code: {result3.returncode}")
    print(f"Response:\n{result3.stdout}")
    print(f"Stderr:\n{result3.stderr}")
    
    # Check for actual tool usage indicators
    tool_indicators = [
        "count_nodes", "database", "nodes", "calling", "querying", 
        "tool", "function", "mcp", "neo4j"
    ]
    
    found_indicators = [ind for ind in tool_indicators if ind.lower() in result3.stdout.lower()]
    print(f"\nTool usage indicators found: {found_indicators}")
    
    if len(found_indicators) >= 2 or "calling" in result3.stdout.lower():
        print("✅ Strong evidence of tool usage!")
    else:
        print("❌ Weak evidence of tool usage")
    
    # Test 4: Explicitly ask Claude to show tool calls
    print("\n" + "="*60)
    print("TEST 4: Ask Claude to show tool invocations")
    print("="*60)
    
    result4 = subprocess.run([
        "claude", "-p",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="Before answering, please show me the exact MCP tool calls you will make. Then use the MCP tools to count nodes in the database and show me the tool invocation details.\n", 
       capture_output=True, text=True, timeout=30)
    
    print(f"\nReturn code: {result4.returncode}")
    print(f"Full Response:\n{result4.stdout}")
    print(f"Stderr:\n{result4.stderr}")
    
    # Check if we can see any tool invocation patterns
    invocation_patterns = ["invoke", "call", "function", "tool", "mcp__", "count_nodes"]
    found_patterns = [p for p in invocation_patterns if p.lower() in result4.stdout.lower()]
    print(f"\nInvocation patterns found: {found_patterns}")
    
    # Test 5: Try JSON output mode to capture tool calls
    print("\n" + "="*60)
    print("TEST 5: JSON output mode to capture tool calls")
    print("="*60)
    
    result5 = subprocess.run([
        "claude", "--output-format", "json",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="Use MCP tools to find the shortest path from Paddington to Tower Hill.\n", 
       capture_output=True, text=True, timeout=30)
    
    print(f"\nReturn code: {result5.returncode}")
    print(f"JSON Response:\n{result5.stdout}")
    
    # Try to parse the JSON to extract tool calls
    try:
        import json
        response_data = json.loads(result5.stdout)
        print(f"\nParsed JSON keys: {list(response_data.keys())}")
        
        # Look for tool calls in the JSON structure
        if 'tool_calls' in response_data:
            print(f"Found tool_calls: {response_data['tool_calls']}")
        elif 'tools' in response_data:
            print(f"Found tools: {response_data['tools']}")
        else:
            print("Searching for tool-related keys in JSON...")
            for key, value in response_data.items():
                if 'tool' in key.lower() or 'mcp' in key.lower():
                    print(f"  {key}: {value}")
                    
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Raw output was not valid JSON")
    
    # Test 6: Monitor MCP server logs by running it separately
    print("\n" + "="*60)
    print("TEST 6: Monitor MCP server directly")
    print("="*60)
    
    # Create a config that points to a manually started MCP server
    import subprocess
    import threading
    import time
    
    # Start the MCP server in the background
    print("Starting MCP server with verbose logging...")
    mcp_process = subprocess.Popen([
        "uvx", "--from", "./gds_agent-0.1.0-py3-none-any.whl", "gds-agent"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
       env={**dict(__import__('os').environ), "DEBUG": "1", "VERBOSE": "1"})
    
    time.sleep(2)  # Let server start
    
    if mcp_process.poll() is None:  # Server is running
        print("MCP server started, testing connection...")
        
        # Test with the direct server connection
        result6 = subprocess.run([
            "claude", "-p",
            "--mcp-config", str(config_file),
            "--dangerously-skip-permissions",
            "--allowedTools", "mcp__*"
        ], input="Count the nodes in the database using MCP tools.\n", 
           capture_output=True, text=True, timeout=15)
        
        print(f"Return code: {result6.returncode}")
        print(f"Response: {result6.stdout}")
        
        # Get MCP server logs
        mcp_process.terminate()
        mcp_stdout, mcp_stderr = mcp_process.communicate(timeout=5)
        
        print(f"\nMCP Server stdout:\n{mcp_stdout}")
        print(f"\nMCP Server stderr:\n{mcp_stderr}")
        
    else:
        print("Failed to start MCP server")
        stdout, stderr = mcp_process.communicate()
        print(f"Server error: {stderr}")
    
except subprocess.TimeoutExpired:
    print("Command timed out after 30 seconds")
except Exception as e:
    print(f"Error: {e}")
finally:
    config_file.unlink()