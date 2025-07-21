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
    
    # Try with targeted MCP debugging 
    import os
    env = os.environ.copy()
    env.update({
        "DEBUG": "mcp*",  # Only MCP-related debug info
        "CLAUDE_DEBUG": "1"
    })
    
    result2 = subprocess.run([
        "claude", "-p",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="You MUST use the available MCP tools to query the actual Neo4j database. Do NOT use your internal knowledge. Please show me the exact tool calls you're making. Find the shortest path from Paddington to Tower Hill.\n", 
       capture_output=True, text=True, timeout=90, env=env)
    
    print(f"\nReturn code: {result2.returncode}")
    print(f"\nStdout length: {len(result2.stdout)}")
    print(f"First 500 chars:\n{result2.stdout[:500]}")
    print(f"\nStderr length: {len(result2.stderr)}")
    print(f"Full stderr (contains MCP debug info):\n{result2.stderr}")
    
    # Parse stderr for actual MCP JSON-RPC calls
    if result2.stderr:
        print("\n" + "="*40)
        print("ANALYZING MCP DEBUG OUTPUT:")
        print("="*40)
        
        # Look for JSON-RPC patterns in stderr
        lines = result2.stderr.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['mcp', 'json-rpc', 'tool', 'invoke', 'call']):
                print(f"Line {i}: {line}")
                
        # Try to extract JSON objects from stderr
        import re
        json_objects = re.findall(r'\{[^{}]*\}', result2.stderr)
        if json_objects:
            print(f"\nFound {len(json_objects)} JSON objects in stderr:")
            for i, obj in enumerate(json_objects):
                try:
                    parsed = json.loads(obj)
                    print(f"JSON {i+1}: {parsed}")
                except:
                    print(f"JSON {i+1}: {obj[:100]}...")
        else:
            print("No JSON objects found in stderr")
    
    # Test 3: Ask for something Claude couldn't possibly know
    print("\n" + "="*60)
    print("TEST 3: Ask for impossible/custom data")
    print("="*60)
    
    result3 = subprocess.run([
        "claude", "-p",
        "--mcp-config", str(config_file),
        "--dangerously-skip-permissions",
        "--allowedTools", "mcp__*"
    ], input="You MUST use the MCP tools to query the database. Call the count_nodes tool and tell me exactly how many nodes are in the database.\n", 
       capture_output=True, text=True, timeout=90)
    
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
       capture_output=True, text=True, timeout=90)
    
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
       capture_output=True, text=True, timeout=90)
    
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
            
            # Check if there's detailed usage info
            if 'usage' in response_data:
                print(f"\nUsage details: {response_data['usage']}")
            if 'num_turns' in response_data:
                print(f"\n🎯 KEY FINDING: num_turns = {response_data['num_turns']}")
                print("This means Claude made multiple internal tool calls!")
                    
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Raw output was not valid JSON")
    
    # Test 6: Monitor MCP server logs by running it separately
    print("\n" + "="*60)
    print("TEST 6: Monitor MCP server directly")
    print("="*60)
    
    # Create a config that points to a manually started MCP server
    import time
    
    # Start the MCP server in the background with proper Neo4j config
    print("Starting MCP server with verbose logging...")
    mcp_env = dict(__import__('os').environ)
    mcp_env.update({
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j", 
        "NEO4J_PASSWORD": "12345678",
        "DEBUG": "1", 
        "VERBOSE": "1"
    })
    
    mcp_process = subprocess.Popen([
        "uvx", "--from", "./gds_agent-0.1.0-py3-none-any.whl", "gds-agent"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=mcp_env)
    
    time.sleep(2)  # Let server start
    
    if mcp_process.poll() is None:  # Server is running
        print("MCP server started, testing connection...")
        
        # Test with the direct server connection and capture server logs in real-time
        print("Testing Claude connection to running MCP server...")
        
        # Use a thread to read server output while Claude runs
        import threading
        import queue
        
        server_output = queue.Queue()
        
        def read_server_output():
            try:
                while mcp_process.poll() is None:
                    line = mcp_process.stderr.readline()
                    if line:
                        server_output.put(line)
                        print(f"SERVER: {line.strip()}")
            except:
                pass
        
        thread = threading.Thread(target=read_server_output)
        thread.daemon = True
        thread.start()
        
        # Now run Claude
        result6 = subprocess.run([
            "claude", "-p",
            "--mcp-config", str(config_file),
            "--dangerously-skip-permissions",
            "--allowedTools", "mcp__*"
        ], input="Count the nodes in the database using MCP tools.\n", 
           capture_output=True, text=True, timeout=30)
        
        print(f"\nClaude return code: {result6.returncode}")
        print(f"Claude response: {result6.stdout}")
        
        # Stop server and collect remaining output
        mcp_process.terminate()
        try:
            mcp_stdout, mcp_stderr = mcp_process.communicate(timeout=5)
            print(f"\nFinal MCP server stdout:\n{mcp_stdout}")
            print(f"\nFinal MCP server stderr:\n{mcp_stderr}")
        except subprocess.TimeoutExpired:
            mcp_process.kill()
            print("Had to force kill MCP server")
        
    else:
        print("Failed to start MCP server")
        stdout, stderr = mcp_process.communicate()
        print(f"Server error: {stderr}")
    
    # Test 7: Try stream-json mode to capture intermediate steps
    print("\n" + "="*60)
    print("TEST 7: Stream JSON mode to capture intermediate steps")
    print("="*60)
    
    try:
        result7 = subprocess.run([
            "claude", "-p", "--verbose", "--output-format", "stream-json",
            "--mcp-config", str(config_file),
            "--dangerously-skip-permissions",
            "--allowedTools", "mcp__*"
        ], input="Use MCP tools to count nodes in the database.\n", 
           capture_output=True, text=True, timeout=90)
        
        print(f"\nReturn code: {result7.returncode}")
        print(f"Stream JSON output:\n{result7.stdout}")
        print(f"Stderr: {result7.stderr}")
        
        # Try to parse each line as JSON to see tool calls
        if result7.stdout:
            print("\nParsing stream JSON lines:")
            for i, line in enumerate(result7.stdout.strip().split('\n')):
                if line.strip():
                    try:
                        data = json.loads(line)
                        print(f"Line {i+1}: {list(data.keys())}")
                        if 'tool' in str(data).lower() or 'mcp' in str(data).lower():
                            print(f"  -> Tool-related content: {data}")
                    except json.JSONDecodeError:
                        print(f"Line {i+1}: Not JSON - {line[:50]}...")
                        
    except subprocess.TimeoutExpired:
        print("Stream JSON mode timed out")
    except Exception as e:
        print(f"Stream JSON mode error: {e}")
    
    # Test 8: Stream JSON for shortest path to capture all tool calls
    print("\n" + "="*60)
    print("TEST 8: Stream JSON for shortest path query")
    print("="*60)
    
    try:
        result8 = subprocess.run([
            "claude", "-p", "--verbose", "--output-format", "stream-json",
            "--mcp-config", str(config_file),
            "--dangerously-skip-permissions",
            "--allowedTools", "mcp__*"
        ], input="Find the shortest path from Paddington to Tower Hill using MCP tools.\n", 
           capture_output=True, text=True, timeout=120)
        
        print(f"\nReturn code: {result8.returncode}")
        print(f"Stream JSON output:\n{result8.stdout}")
        print(f"Stderr: {result8.stderr}")
        
        # Parse and analyze the tool calls for shortest path
        if result8.stdout:
            print("\n" + "="*50)
            print("ANALYZING SHORTEST PATH TOOL CALLS:")
            print("="*50)
            
            for i, line in enumerate(result8.stdout.strip().split('\n')):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # Look for tool calls
                        if data.get('type') == 'assistant' and 'message' in data:
                            content = data['message'].get('content', [])
                            for item in content:
                                if item.get('type') == 'tool_use':
                                    print(f"🔧 TOOL CALL: {item['name']}")
                                    print(f"   Parameters: {item.get('input', {})}")
                                    print(f"   ID: {item['id']}")
                                    
                        # Look for tool results
                        elif data.get('type') == 'user' and 'message' in data:
                            content = data['message'].get('content', [])
                            for item in content:
                                if item.get('type') == 'tool_result':
                                    result_text = item.get('content', [{}])[0].get('text', 'No result')
                                    print(f"📥 TOOL RESULT (ID: {item['tool_use_id']}): {result_text[:200]}...")
                                    
                        # Look for final summary
                        elif data.get('type') == 'result':
                            print(f"✅ FINAL RESULT:")
                            print(f"   Turns: {data.get('num_turns', 'Unknown')}")
                            print(f"   Duration: {data.get('duration_ms', 'Unknown')}ms")
                            print(f"   Cost: ${data.get('total_cost_usd', 'Unknown')}")
                            
                    except json.JSONDecodeError:
                        pass  # Skip non-JSON lines
                        
    except subprocess.TimeoutExpired:
        print("Shortest path stream JSON timed out")
    except Exception as e:
        print(f"Shortest path stream JSON error: {e}")
    
except subprocess.TimeoutExpired:
    print("Command timed out after 30 seconds")
except Exception as e:
    print(f"Error: {e}")
finally:
    config_file.unlink()