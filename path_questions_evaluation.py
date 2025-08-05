import re
import ast
from typing import List, Dict, Tuple, Any, Optional

# Map algorithm tools to their output format types
ALGORITHM_FORMATS = {
    # Path with costs format: ["station1", "station2"]: [cost1, cost2]
    'delta_stepping_shortest_path': 'path_with_costs',
    'find_shortest_path': 'path_with_costs', 
    'dijkstra_single_source_shortest_path': 'path_with_costs',
    'a_star_shortest_path': 'path_with_costs',
    'yens_shortest_paths': 'path_with_costs',
    'bellman_ford_single_source_shortest_path': 'path_with_costs',
    'longest_path': 'path_with_costs',
    
    # Tree tuples format: ("node", "parent", weight)
    'minimum_weight_spanning_tree': 'tree_tuples',
    'minimum_directed_steiner_tree': 'tree_tuples',
    'prize_collecting_steiner_tree': 'tree_tuples',
    
    # Distance tuples format: ("source", "target", distance)
    'all_pairs_shortest_paths': 'distance_tuples',
    
    # Simple paths format: ["A", "B", "C"]
    'random_walk': 'simple_paths',
    'breadth_first_search': 'simple_paths',
    'depth_first_search': 'simple_paths',
}

def extract_algorithm_type(expected_tools: str) -> Optional[str]:
    """
    Extract the main algorithm from expected tools list
    Expected tools format: ["mcp__gds-agent__get_node_properties", "mcp__gds-agent__dijkstra"]
    """
    try:
        tools_list = ast.literal_eval(expected_tools)
        
        # Filter out non-algorithm tools
        non_algorithm_tools = {
            'mcp__gds-agent__get_node_properties',
            'mcp__gds-agent__get_relationship_properties',
            'mcp__gds-agent__get_node_properties_keys',
            'mcp__gds-agent__get_relationship_properties_keys'
        }
        
        # Find the algorithm tool
        for tool in tools_list:
            if tool.startswith('mcp__gds-agent__') and tool not in non_algorithm_tools:
                # Extract algorithm name (remove prefix)
                algorithm_name = tool.replace('mcp__gds-agent__', '')
                return algorithm_name
        
        return None
    except (ValueError, SyntaxError):
        return None

def evaluate_path_algorithm_output(expected_tools: str, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
    """
    Evaluate algorithm output based on expected tools and answers
    """
    # Extract algorithm type
    algorithm_type = extract_algorithm_type(expected_tools)
    if not algorithm_type:
        return {"error": "Could not determine algorithm type"}
    
    # Get format type
    format_type = ALGORITHM_FORMATS.get(algorithm_type)
    if not format_type:
        return {"error": f"Unknown algorithm type: {algorithm_type}"}
    
    # Evaluate based on format type
    if format_type == 'path_with_costs':
        return evaluate_path_with_costs(expected_answer, actual_answer, algorithm_type)
    elif format_type == 'tree_tuples':
        return evaluate_tree_tuples(expected_answer, actual_answer, algorithm_type)
    elif format_type == 'distance_tuples':
        return evaluate_distance_tuples(expected_answer, actual_answer, algorithm_type)
    elif format_type == 'simple_paths':
        return evaluate_simple_paths(expected_answer, actual_answer, algorithm_type)
    else:
        return {"error": f"Unknown format type: {format_type}"}
    
def evaluate_path_with_costs(expected: str, actual: str, algorithm_type: str) -> Dict[str, Any]:
    """
    Evaluate path with costs format: ["station1", "station2"]: [cost1, cost2]
    """
    # Regex pattern to match: ["path"]: [costs]
    # Handle escaped quotes and various spacing
    pattern = r'\[\"([^\"\\]*(?:\\.[^\"\\]*)*(?:\",\s*\"[^\"\\]*(?:\\.[^\"\\]*)*)*)\"\]:\s*\[([0-9.]+(?:,\s*[0-9.]+)*)\]'
    
    def extract_path_costs(text: str) -> List[Tuple[List[str], List[float]]]:
        matches = re.findall(pattern, text)
        results = []
        for path_str, costs_str in matches:
            # Parse path stations - handle escaped quotes
            path_stations = []
            # Split by '", "' but handle escaped quotes
            parts = re.split(r'",\s*"', path_str)
            for part in parts:
                # Remove leading/trailing quotes and unescape
                station = part.strip().strip('"').replace('\\"', '"').replace("\\'", "'")
                path_stations.append(station)
            
            # Parse costs
            costs = [float(cost.strip()) for cost in costs_str.split(',')]
            results.append((path_stations, costs))
        return results
    
    try:
        expected_paths = extract_path_costs(expected)
        actual_paths = extract_path_costs(actual)
        
        # For single path algorithms (dijkstra, a_star, longest_path)
        if algorithm_type in ['find_shortest_path', 'a_star_shortest_path', 'longest_path']:
            if len(expected_paths) != 1 or len(actual_paths) != 1:
                return {
                    "success": False,
                    "error": "Expected single path",
                    "expected_count": len(expected_paths),
                    "actual_count": len(actual_paths)
                }
            
            exp_path, exp_costs = expected_paths[0]
            act_path, act_costs = actual_paths[0]
            
            path_match = exp_path == act_path
            costs_match = exp_costs == act_costs
            
            return {
                "success": path_match and costs_match,
                "path_match": path_match,
                "costs_match": costs_match,
                "expected_path": exp_path,
                "actual_path": act_path,
                "expected_costs": exp_costs,
                "actual_costs": act_costs
            }
        
        # For multi-path algorithms (delta_stepping, dijkstra_single_source, yens, bellman_ford)
        else:
            # Convert to sets for comparison (order may vary)
            expected_set = set((tuple(path), tuple(costs)) for path, costs in expected_paths)
            actual_set = set((tuple(path), tuple(costs)) for path, costs in actual_paths)
            
            return {
                "success": expected_set == actual_set,
                "expected_count": len(expected_paths),
                "actual_count": len(actual_paths),
                "missing_paths": list(expected_set - actual_set),
                "extra_paths": list(actual_set - expected_set),
                "matching_paths": list(expected_set & actual_set)
            }
            
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {str(e)}"}

def evaluate_tree_tuples(expected: str, actual: str, algorithm_type: str) -> Dict[str, Any]:
    """
    Evaluate tree tuples format: ("node", "parent", weight)
    """
    # Regex pattern to match: ("node", "parent", weight)
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*([0-9.]+)\)'
    
    def extract_tree_tuples(text: str) -> List[Tuple[str, str, float]]:
        matches = re.findall(pattern, text)
        return [(node.strip(), parent.strip(), float(weight)) for node, parent, weight in matches]
    
    try:
        expected_tuples = extract_tree_tuples(expected)
        actual_tuples = extract_tree_tuples(actual)
        
        # Convert to sets for comparison
        expected_set = set(expected_tuples)
        actual_set = set(actual_tuples)
        
        return {
            "success": expected_set == actual_set,
            "expected_count": len(expected_tuples),
            "actual_count": len(actual_tuples),
            "missing_tuples": list(expected_set - actual_set),
            "extra_tuples": list(actual_set - expected_set),
            "matching_tuples": list(expected_set & actual_set)
        }
        
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {str(e)}"}

def evaluate_distance_tuples(expected: str, actual: str, algorithm_type: str) -> Dict[str, Any]:
    """
    Evaluate distance tuples format: ("source", "target", distance)
    """
    # Regex pattern to match: ("source", "target", distance)
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*([0-9.]+)\)'
    
    def extract_distance_tuples(text: str) -> List[Tuple[str, str, float]]:
        matches = re.findall(pattern, text)
        return [(source.strip(), target.strip(), float(distance)) for source, target, distance in matches]
    
    try:
        expected_tuples = extract_distance_tuples(expected)
        actual_tuples = extract_distance_tuples(actual)
        
        # Convert to sets for comparison
        expected_set = set(expected_tuples)
        actual_set = set(actual_tuples)
        
        return {
            "success": expected_set == actual_set,
            "expected_count": len(expected_tuples),
            "actual_count": len(actual_tuples),
            "missing_distances": list(expected_set - actual_set),
            "extra_distances": list(actual_set - expected_set),
            "matching_distances": list(expected_set & actual_set)
        }
        
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {str(e)}"}

def evaluate_simple_paths(expected: str, actual: str, algorithm_type: str) -> Dict[str, Any]:
    """
    Evaluate simple paths format: ["A", "B", "C"], ["D", "A", "F"]
    """
    # Regex pattern to match: ["station1", "station2", "station3"]
    pattern = r'\[\"([^\"\\]*(?:\\.[^\"\\]*)*(?:\",\s*\"[^\"\\]*(?:\\.[^\"\\]*)*)*)\"\]'
    
    def extract_simple_paths(text: str) -> List[List[str]]:
        matches = re.findall(pattern, text)
        paths = []
        for match in matches:
            # Parse path stations - handle escaped quotes
            path_stations = []
            parts = re.split(r'",\s*"', match)
            for part in parts:
                station = part.strip().strip('"').replace('\\"', '"').replace("\\'", "'")
                path_stations.append(station)
            paths.append(path_stations)
        return paths
    
    try:
        expected_paths = extract_simple_paths(expected)
        actual_paths = extract_simple_paths(actual)
        
        # For BFS/DFS (typically single path)
        if algorithm_type in ['breadth_first_search', 'depth_first_search']:
            if len(expected_paths) != 1 or len(actual_paths) != 1:
                return {
                    "success": False,
                    "error": "Expected single path",
                    "expected_count": len(expected_paths),
                    "actual_count": len(actual_paths)
                }
            
            return {
                "success": expected_paths[0] == actual_paths[0],
                "expected_path": expected_paths[0],
                "actual_path": actual_paths[0]
            }
        
        # For random walks (multiple paths)
        else:
            # Convert to sets for comparison
            expected_set = set(tuple(path) for path in expected_paths)
            actual_set = set(tuple(path) for path in actual_paths)
            
            return {
                "success": expected_set == actual_set,
                "expected_count": len(expected_paths),
                "actual_count": len(actual_paths),
                "missing_paths": [list(path) for path in (expected_set - actual_set)],
                "extra_paths": [list(path) for path in (actual_set - expected_set)],
                "matching_paths": [list(path) for path in (expected_set & actual_set)]
            }
            
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {str(e)}"}

# Test function
def test_evaluator():
    """Test the evaluator with sample data"""
    # Test path with costs
    print("Testing path with costs...")
    expected = '["Barons Court", "Earl\'s Court", "Gloucester Road", "South Kensington"]: [0.0, 1.0, 2.0, 3.0]'
    actual = '["Barons Court", "Earl\'s Court", "Gloucester Road", "South Kensington"]: [0.0, 1.0, 2.0, 3.0]'
    result = evaluate_path_with_costs(expected, actual, 'find_shortest_path')
    print(f"Result: {result}")
    
    # Test tree tuples
    print("\nTesting tree tuples...")
    expected = '("Bank", "Bank", 0.0), ("London Bridge", "Bank", 2.0), ("Southwark", "London Bridge", 2.0)'
    actual = '("Bank", "Bank", 0.0), ("London Bridge", "Bank", 2.0), ("Southwark", "London Bridge", 2.0)'
    result = evaluate_tree_tuples(expected, actual, 'minimum_directed_steiner_tree')
    print(f"Result: {result}")
    
    # Test simple paths
    print("\nTesting simple paths...")
    expected = '["Moorgate", "Old Street"], ["Farringdon", "King\'s Cross St. Pancras", "Russell Square"]'
    actual = '["Moorgate", "Old Street"], ["Farringdon", "King\'s Cross St. Pancras", "Russell Square"]'
    result = evaluate_simple_paths(expected, actual, 'random_walk')
    print(f"Result: {result}")
    
    # Test algorithm extraction
    print("\nTesting algorithm extraction...")
    tools = '["mcp__gds-agent__get_node_properties", "mcp__gds-agent__dijkstra_single_source_shortest_path"]'
    algorithm = extract_algorithm_type(tools)
    print(f"Extracted algorithm: {algorithm}")
    
    # Test full evaluation
    print("\\nTesting full path algorithm evaluation...")
    result = evaluate_path_algorithm_output(tools, expected, actual)
    print(f"Full evaluation result: {result}")

if __name__ == "__main__":
    test_evaluator()