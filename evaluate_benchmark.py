import json
import logging
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob
from path_questions_evaluation import evaluate_path_algorithm_output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WORKFLOW_TOOLS = {
    "mcp__gds-agent__project_graph_cypher",
    "mcp__gds-agent__list_graphs",
    "mcp__gds-agent__drop_graph",
    "mcp__gds-agent__list_sessions",
    "mcp__gds-agent__create_session",
    "mcp__gds-agent__delete_session",
}

TOOL_EQUIVALENTS = {
    "mcp__gds-agent__count_nodes": {
        "mcp__gds-agent__count_nodes",
        "mcp__gds-agent__get_graph_info",
    },
}


class BenchmarkEvaluator:
    def __init__(self, dataset: str = "ln", model: str = "sonnet-4-20250514", questions_file: Optional[Path] = None):
        # Map dataset shorthand names to question files
        dataset_files = {
            "ln": "questions/gds-algo-questions-ln.csv",
            "got": "questions/gds-algo-questions-got.csv"
        }

        self.dataset = dataset
        if dataset in dataset_files:
            self.questions_file = Path(questions_file) if questions_file is not None else Path(dataset_files[dataset])
        else:
            if questions_file is None:
                raise ValueError(
                    f"Unknown dataset '{dataset}': --questions-file is required for custom datasets."
                )
            self.questions_file = Path(questions_file)

        self.model = model
        
        # Look for results files matching the dataset and model pattern
        pattern = f"results_{model}/{dataset}_results_*.json"
        self.results_files = list(Path().glob(pattern))

        # Set evaluation output file with dataset and model-specific name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluation_file = Path(f"results_{model}/{dataset}_evaluation_{timestamp}.json")
        self.question_trace_report_file = Path(f"results_{model}/{dataset}_question_trace_report_{timestamp}.json")
        
    def load_expected_results(self) -> Dict[str, Dict[str, Any]]:
        """Load expected results from a questions file.

        Supports JSON (array of objects with 'Question' and 'Expected answer')
        and CSV (4-lines-per-question format).
        """
        expected = {}
        
        if not self.questions_file.exists():
            logger.error(f"Questions file not found: {self.questions_file}")
            return expected
            
        try:
            if self.questions_file.suffix == ".json":
                with open(self.questions_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                question_id = 1
                for entry in data:
                    question = (entry.get("Question") or "").strip()
                    if not question:
                        continue
                    answer = entry.get("Expected answer", "")
                    qid = entry.get("id", question_id)
                    expected[question] = {
                        'id': qid,
                        'expected_tools': entry.get("expected_tools", []),
                        'expected_parameters': entry.get("expected_parameters", {}),
                        'expected_answer': answer if isinstance(answer, str) else str(answer)
                    }
                    question_id += 1
            else:
                with open(self.questions_file, 'r', encoding='utf-8') as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]
                    
                    if not lines:
                        logger.error("Questions file is empty")
                        return expected
                    
                    # Process lines in groups of 4: question, tools, parameters, answer
                    i = 1
                    question_id = 1
                    while i + 3 < len(lines):
                        question = lines[i].strip()
                        tools_str = lines[i+1].strip()
                        params_str = lines[i+2].strip()
                        answer = lines[i+3].strip()
                        
                        if question:
                            try:
                                expected[question] = {
                                    'id': question_id,
                                    'expected_tools': json.loads(tools_str),
                                    'expected_parameters': json.loads(params_str),
                                    'expected_answer': answer
                                }
                                question_id += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON for question: {question[:50]}...")
                                logger.warning(f"Tools string: '{tools_str}'")
                                logger.warning(f"Params string: '{params_str}'")
                                logger.warning(f"JSON Error: {e}")
                        
                        i += 4
                    
        except Exception as e:
            logger.error(f"Error loading expected results: {e}")
            
        logger.info(f"Loaded {len(expected)} expected results")
        return expected
        
    def extract_token_usage_from_raw_stream(self, raw_stream: str) -> Dict[str, Any]:
        import json as json_module
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0
        total_cost_usd = 0.0
        message_count = 0
        
        if not raw_stream:
            return {}
            
        try:
            lines = raw_stream.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        message = json_module.loads(line)
                        
                        if message.get('type') == 'assistant' and 'message' in message:
                            usage = message['message'].get('usage', {})
                            if usage:
                                total_input_tokens += usage.get('input_tokens', 0)
                                total_output_tokens += usage.get('output_tokens', 0)
                                total_cache_creation_tokens += usage.get('cache_creation_input_tokens', 0)
                                total_cache_read_tokens += usage.get('cache_read_input_tokens', 0)
                                message_count += 1
                        
                        elif message.get('type') == 'result' and 'total_cost_usd' in message:
                            total_cost_usd = message.get('total_cost_usd', 0.0)
                            final_usage = message.get('usage', {})
                            if final_usage:
                                total_input_tokens = final_usage.get('input_tokens', total_input_tokens)
                                total_output_tokens = final_usage.get('output_tokens', total_output_tokens)
                                total_cache_creation_tokens = final_usage.get('cache_creation_input_tokens', total_cache_creation_tokens)
                                total_cache_read_tokens = final_usage.get('cache_read_input_tokens', total_cache_read_tokens)
                            
                    except json_module.JSONDecodeError:
                        logger.error(f"Skipped malformed JSON line: {line[:100] if line else 'None'}...")
                        
        except Exception as e:
            logger.warning(f"Error parsing raw_stream for token usage: {e}")
            
        return {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cache_creation_tokens': total_cache_creation_tokens,
            'total_cache_read_tokens': total_cache_read_tokens,
            'total_cost_usd': total_cost_usd,
            'message_count': message_count,
            'total_tokens': total_input_tokens + total_output_tokens + total_cache_creation_tokens
        }

    def load_actual_results(self, include_failed: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        all_results = {}
        
        if not self.results_files:
            logger.error("No results files found")
            return all_results
            
        logger.info(f"Loading results from {len(self.results_files)} files")
        
        for results_file in self.results_files:
            if not results_file.exists():
                logger.warning(f"Results file not found: {results_file}")
                continue
                
            try:
                with open(results_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                logger.info(f"Processing {results_file}")
                
                for result in data.get('raw_results', []):
                    if result.get('response_data') and (include_failed or result.get('success', False)):
                        question = result['question'].strip()
                        response_data = result['response_data']
                        
                        token_usage = self.extract_token_usage_from_raw_stream(
                            response_data.get('raw_stream', '')
                        )
                        
                        result_data = {
                            'tool_calls': response_data.get('tool_calls', []),
                            'tool_results': response_data.get('tool_results', []),
                            'final_result': response_data.get('final_result', ''),
                            'num_turns': response_data.get('num_turns', 0),
                            'duration_ms': response_data.get('duration_ms', 0),
                            'token_usage': token_usage,
                            'source_file': str(results_file)
                        }
                        
                        if question not in all_results:
                            all_results[question] = []
                        all_results[question].append(result_data)
                        
            except Exception as e:
                logger.error(f"Error loading actual results from {results_file}: {e}")
                
        total_results = sum(len(runs) for runs in all_results.values())
        logger.info(f"Loaded {total_results} total results across {len(all_results)} questions")
        return all_results
        
    def evaluate_tool_calls(self, expected_tools: List[str], actual_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if the correct tools were called."""
        def matches(expected_name: str, actual_name: str) -> bool:
            return actual_name in TOOL_EQUIVALENTS.get(
                expected_name, {expected_name}
            )

        actual_tool_names = [tool['name'] for tool in actual_tools]
        scored_actual_tool_names = [
            name
            for name in actual_tool_names
            if name not in WORKFLOW_TOOLS
            or any(matches(expected, name) for expected in expected_tools)
        ]

        missing_tools = [
            expected
            for expected in expected_tools
            if not any(
                matches(expected, actual) for actual in scored_actual_tool_names
            )
        ]
        unexpected_tools = [
            actual
            for actual in scored_actual_tool_names
            if not any(matches(expected, actual) for expected in expected_tools)
        ]
        
        if len(expected_tools) == 0:
            precision = 1.0 if len(scored_actual_tool_names) == 0 else 0.0
            recall = 1.0
        else:
            unique_correct_tools = sum(
                any(matches(expected, actual) for actual in scored_actual_tool_names)
                for expected in set(expected_tools)
            )
            
            recall = unique_correct_tools / len(set(expected_tools))
            
            total_actual_calls = len(scored_actual_tool_names)
            precision = unique_correct_tools / total_actual_calls if total_actual_calls > 0 else 1.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'missing_tools': missing_tools,
            'unexpected_tools': unexpected_tools,
            'workflow_tools': [
                name for name in actual_tool_names if name in WORKFLOW_TOOLS
            ],
            'exact_match': len(missing_tools) == 0 and len(unexpected_tools) == 0
        }
    
    def _values_equal(self, expected_value: Any, actual_value: Any) -> bool:
        """Compare two values, treating lists as equal regardless of order."""
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            # For lists, compare as sets (order doesn't matter)
            return set(expected_value) == set(actual_value)
        else:
            # For non-lists, use regular comparison
            return actual_value == expected_value or str(actual_value) == str(expected_value)
        
    def evaluate_parameters(self, expected_params: Dict[str, Any], actual_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if the correct parameters were used."""
        parameter_scores = {}
        
        for tool_key, expected_tool_params in expected_params.items():
            matching_tools = []
            for tool in actual_tools:
                if tool_key.lower() in tool['name'].lower():
                    matching_tools.append(tool)
            
            if not matching_tools:
                parameter_scores[tool_key] = {
                    'match': False,
                    'reason': 'Tool not called'
                }
                continue
                
            actual_params = matching_tools[0].get('parameters', {})
            ignored_parameters = []
            parameters_to_score = expected_tool_params
            if "graphName" in actual_params:
                ignored_parameters = [
                    key
                    for key in ("nodeLabels", "relTypes")
                    if key in expected_tool_params
                ]
                parameters_to_score = {
                    key: value
                    for key, value in expected_tool_params.items()
                    if key not in ignored_parameters
                }
            
            matches = []
            mismatches = []
            
            for param_key, expected_value in parameters_to_score.items():
                actual_value = actual_params.get(param_key)

                # Handle range constraints like "<=5"
                if isinstance(expected_value, str) and expected_value.startswith('<='):
                    try:
                        max_value = int(expected_value[2:])
                        if actual_value is not None and int(actual_value) <= max_value:
                            matches.append(param_key)
                        else:
                            mismatches.append({
                                'param': param_key,
                                'expected': expected_value,
                                'actual': actual_value
                            })
                    except (ValueError, TypeError):
                        mismatches.append({
                            'param': param_key,
                            'expected': expected_value,
                            'actual': actual_value
                        })
                elif self._values_equal(expected_value, actual_value):
                    matches.append(param_key)
                else:
                    mismatches.append({
                        'param': param_key,
                        'expected': expected_value,
                        'actual': actual_value
                    })
            
            parameter_scores[tool_key] = {
                'match': len(mismatches) == 0,
                'matches': matches,
                'mismatches': mismatches,
                'ignored_parameters': ignored_parameters,
                'score': len(matches) / len(parameters_to_score) if parameters_to_score else 1.0
            }
            
        return parameter_scores
        
    def evaluate_answer_similarity(self, expected_answer: str, actual_answer: str, expected_tools: List[str] = None) -> Dict[str, Any]:        
        if "path_questions" in str(self.questions_file) and expected_tools:
            result = evaluate_path_algorithm_output(str(expected_tools), expected_answer, actual_answer)
            
            if result.get('success'):
                return {
                    'answer_match_score': 1.0,
                    'answer_matched_count': 1,
                    'answer_total_count': 1
                }
            else:
                return {
                    'answer_match_score': 0.0,
                    'answer_matched_count': 0,
                    'answer_total_count': 1
                }
        else:
            return self.evaluate_unified_answer(expected_answer, actual_answer)
    
    def evaluate_unified_answer(self, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        import re
        
        def normalize_text(text):
            return text.lower().strip()
        
        # Handle empty expected answer (e.g., longest path with no result)
        if not expected_answer.strip():
            # Expected answer is empty, actual should also be empty
            if not actual_answer.strip():
                return {
                    'answer_match_score': 1.0,
                    'answer_matched_count': 1,
                    'answer_total_count': 1
                }
            else:
                return {
                    'answer_match_score': 0.0,
                    'answer_matched_count': 0,
                    'answer_total_count': 1
                }
        
        # Handle Yes/No questions
        expected_normalized = normalize_text(expected_answer)
        actual_normalized = normalize_text(actual_answer)
        
        if expected_normalized in ['yes', 'no']:
            # Simple Yes/No comparison
            if expected_normalized == actual_normalized:
                return {
                    'answer_match_score': 1.0,
                    'answer_matched_count': 1,
                    'answer_total_count': 1
                }
            else:
                return {
                    'answer_match_score': 0.0,
                    'answer_matched_count': 0,
                    'answer_total_count': 1
                }
        
        def extract_structured_items(text):
            items = set()
            
            # Pattern 1: "path: cost" format (e.g., "path1: 5.0, path2: 3.2")
            path_cost_pattern = r'([^:,]+):\s*([0-9.]+)'
            path_cost_matches = re.findall(path_cost_pattern, text)
            for path, cost in path_cost_matches:
                # Normalize path (remove extra spaces, convert to lowercase)
                normalized_path = re.sub(r'\s+', ' ', path.strip().lower())
                items.add((normalized_path, float(cost)))
            
            # Pattern 2: "(node, parent, weight)" format
            tuple_pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([0-9.]+)\s*\)'
            tuple_matches = re.findall(tuple_pattern, text)
            for node, parent, weight in tuple_matches:
                items.add((normalize_text(node), normalize_text(parent), float(weight)))
            
            # Pattern 3: "station:score" format (centrality results)
            station_score_pattern = r'([A-Za-z\s\(\)]+):\s*([0-9.]+)'
            station_score_matches = re.findall(station_score_pattern, text)
            for station, score in station_score_matches:
                normalized_station = normalize_text(station)
                items.add((normalized_station, float(score)))
            
            # Pattern 4: JSON list format like ["A", "B", "C"]
            if not items:
                try:
                    import json
                    # Try to parse as JSON list
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        for item in parsed:
                            items.add(normalize_text(str(item)))
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Pattern 5: Simple comma-separated paths/items (if no other patterns found)
            if not items:
                # Split by common separators and clean up
                simple_items = re.split(r'[,;|]', text)
                for item in simple_items:
                    cleaned_item = normalize_text(item)
                    if cleaned_item:
                        items.add(cleaned_item)
            
            # Pattern 6: Numbers only (counts, etc.)
            if not items:
                number_pattern = r'\b(\d+(?:\.\d+)?)\b'
                numbers = re.findall(number_pattern, text)
                for num in numbers:
                    items.add(float(num))
            
            return items
        
        expected_items = extract_structured_items(expected_answer)
        actual_items = extract_structured_items(actual_answer)
        
        # Calculate match score (order-independent)
        if len(expected_items) == 0:
            # If no expected items, only match if actual is also empty
            answer_match_score = 1.0 if len(actual_items) == 0 else 0.0
            matched_count = 1 if len(actual_items) == 0 else 0
            total_expected_count = 1
        else:
            # Calculate how many expected items are found in actual
            matched_count = len(expected_items.intersection(actual_items))
            
            # For k-shortest paths: if actual is a subset of expected, that's acceptable
            if len(actual_items) > 0 and actual_items.issubset(expected_items):
                # All actual items are valid (subset of expected)
                answer_match_score = 1.0
                total_expected_count = len(actual_items)
                matched_count = len(actual_items)
            else:
                # Standard evaluation: how many expected items are found
                total_expected_count = len(expected_items)
                answer_match_score = matched_count / total_expected_count
        
        return {
            'answer_match_score': answer_match_score,
            'answer_matched_count': matched_count,
            'answer_total_count': total_expected_count
        }
    
    def evaluate_centrality_answer(self, expected_answer: str, actual_answer: str) -> Dict[str, Any]:        
        def extract_result_scores(text):
            # Pattern to match "Name: number" or "Name**: number" etc.
            score_pattern = r'([A-Za-z\s\(\)]+)[:*\s]+(\d*\.?\d+)'
            matches = re.findall(score_pattern, text)
            
            scores = {}
            for name, score in matches:
                name = re.sub(r'[*\-\s]+', ' ', name).strip()
                name = re.sub(r'\s+', ' ', name)
                scores[name.lower()] = score
            
            return scores
            
        def extract_score_numbers(text):
            return re.findall(r'\b0\.\d+\b', text)
            
        expected_scores = extract_result_scores(expected_answer)
        actual_scores = extract_result_scores(actual_answer)
        print("answers:",expected_answer,", ", actual_answer)
        print("scores:",expected_scores,", ", actual_scores)
        # Fallback to just score numbers if name matching fails
        expected_numbers = extract_score_numbers(expected_answer)
        actual_numbers = extract_score_numbers(actual_answer)
        
        # Check for exact score matches
        exact_score_matches = 0
        total_expected_scores = len(expected_scores) if expected_scores else len(expected_numbers)
        
        if expected_scores and actual_scores:
            # Compare named scores
            for expected_name, expected_score in expected_scores.items():
                for actual_name, actual_score in actual_scores.items():
                    # Fuzzy name matching for stations like "Paddington" vs "paddington"
                    if expected_name in actual_name or actual_name in expected_name:
                        score_float = float(expected_score)
                        actual_float = float(actual_score)
                        if abs(actual_float-score_float) <=1e-5:
                            exact_score_matches += 1
                        break
        else:
            # Fallback to number-only comparison
            exact_score_matches = sum(1 for num in expected_numbers if num in actual_numbers)
        
        # Calculate score-based accuracy
        number_match_score = exact_score_matches / total_expected_scores if total_expected_scores > 0 else 0.0
        
        # Simple key content check - does actual contain expected station names
        def extract_station_names(text):
            # Common patterns for station names
            stations = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*(?:\s*\([A-Z]\))?)\b', text)
            return [s[0].lower() if isinstance(s, tuple) else s.lower() for s in stations]
        
        expected_stations = set(extract_station_names(expected_answer))
        actual_stations = set(extract_station_names(actual_answer))
        
        station_coverage = len(expected_stations.intersection(actual_stations)) / len(expected_stations) if expected_stations else 1.0
        
        return {
            'exact_number_matches': exact_score_matches,
            'total_expected_numbers': total_expected_scores,
            'number_match_score': number_match_score,
            'station_coverage': station_coverage,
            'exact_match': number_match_score == 1.0 and station_coverage >= 0.8,
            'expected_scores': expected_scores,
            'actual_scores': actual_scores,
            'expected_numbers': expected_numbers,
            'actual_numbers': actual_numbers
        }
        
    def evaluate_question(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        expected_tools = expected.get('expected_tools', [])
        expected_parameters = expected.get('expected_parameters', {})

        tool_evaluation = self.evaluate_tool_calls(
            expected_tools,
            actual['tool_calls']
        )

        parameter_evaluation = self.evaluate_parameters(
            expected_parameters,
            actual['tool_calls']
        )

        answer_evaluation = self.evaluate_answer_similarity(
            expected['expected_answer'],
            actual['final_result'],
            expected_tools
        )
        
        tool_score = tool_evaluation['f1_score']
        param_score = sum(p.get('score', 0.0) for p in parameter_evaluation.values()) / len(parameter_evaluation) if parameter_evaluation else 0.0
        answer_score = answer_evaluation['answer_match_score']

        if expected_tools or expected_parameters:
            overall_score = (tool_score + param_score + answer_score) / 3
        else:
            overall_score = answer_score
        
        return {
            'overall_score': overall_score,
            'tool_evaluation': tool_evaluation,
            'parameter_evaluation': parameter_evaluation,
            'answer_evaluation': answer_evaluation,
            'metadata': {
                'num_turns': actual.get('num_turns', 0),
                'duration_ms': actual.get('duration_ms', 0)
            },
            'token_usage': actual.get('token_usage', {})
        }
        
    def run_evaluation(self) -> Dict[str, Any]:
        logger.info("Starting benchmark evaluation...")
        
        expected_results = self.load_expected_results()
        actual_results = self.load_actual_results()
        
        if not expected_results:
            logger.error("No expected results loaded")
            return {}
            
        if not actual_results:
            logger.error("No actual results loaded")
            return {}
            
        evaluations = {}
        aggregated_stats = {}
        
        for question in expected_results:
            if question in actual_results:
                logger.info(f"Evaluating: {question[:50]}... ({len(actual_results[question])} runs)")
                
                run_evaluations = []
                for i, actual_result in enumerate(actual_results[question]):
                    run_eval = self.evaluate_question(
                        expected_results[question],
                        actual_result
                    )
                    run_eval['run_number'] = i + 1
                    run_eval['source_file'] = actual_result.get('source_file', '')
                    run_evaluations.append(run_eval)
                
                evaluations[question] = {
                    'runs': run_evaluations,
                    'num_runs': len(run_evaluations)
                }
                
                scores = [run['overall_score'] for run in run_evaluations]
                tool_f1s = [run['tool_evaluation']['f1_score'] for run in run_evaluations]
                answer_scores = [run['answer_evaluation']['answer_match_score'] for run in run_evaluations]
                
                aggregated_stats[question] = {
                    'overall_score_mean': sum(scores) / len(scores),
                    'overall_score_min': min(scores),
                    'overall_score_max': max(scores),
                    'tool_f1_mean': sum(tool_f1s) / len(tool_f1s),
                    'answer_score_mean': sum(answer_scores) / len(answer_scores),
                    'success_rate': len([s for s in scores if s > 0.8]) / len(scores),
                    'num_runs': len(scores)
                }
            else:
                logger.warning(f"No actual result found for question: {question[:50]}...")
                evaluations[question] = {
                    'runs': [],
                    'num_runs': 0,
                    'error': 'No actual result found'
                }
                aggregated_stats[question] = {
                    'overall_score_mean': 0.0,
                    'success_rate': 0.0,
                    'num_runs': 0
                }
        
        all_mean_scores = [stats['overall_score_mean'] for stats in aggregated_stats.values() if stats['num_runs'] > 0]
        total_runs = sum(stats['num_runs'] for stats in aggregated_stats.values())
        
        summary = {
            'dataset': self.dataset,
            'model': self.model,
            'questions_file': str(self.questions_file),
            'total_questions': len(expected_results),
            'total_runs': total_runs,
            'average_score_across_questions': sum(all_mean_scores) / len(all_mean_scores) if all_mean_scores else 0.0,
            'results_files_processed': len(self.results_files)
        }
        
        return {
            'summary': summary,
            'aggregated_stats': aggregated_stats,
            'detailed_evaluations': evaluations
        }

    def _join_tool_traces(self, tool_calls: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results_by_id = {
            result.get("tool_use_id"): result.get("result", "")
            for result in tool_results
            if result.get("tool_use_id")
        }
        joined = []
        for index, call in enumerate(tool_calls):
            call_id = call.get("id") or ""
            if call_id in results_by_id:
                result = results_by_id[call_id]
            elif not call_id and index < len(tool_results):
                result = tool_results[index].get("result", "")
            else:
                result = ""
            joined.append({
                "id": call_id,
                "name": call.get("name", ""),
                "parameters": call.get("parameters", {}),
                "result": result,
            })
        return joined

    def write_question_trace_report(self, path: Path) -> None:
        """Write a per-question JSON of expected vs agent answers, scores, and tool traces."""
        expected_results = self.load_expected_results()
        actual_results = self.load_actual_results(include_failed=True)
        report = []

        for question, expected in expected_results.items():
            runs = actual_results.get(question, [])
            if not runs:
                report.append({
                    "question_id": expected.get("id"),
                    "question": question,
                    "expected_answer": expected.get("expected_answer", ""),
                    "agent_answer": None,
                    "overall_score": None,
                    "answer_score": None,
                    "answer_matched_count": None,
                    "answer_total_count": None,
                    "tool_f1": None,
                    "source_file": None,
                    "tools": [],
                })
                continue
            for run in runs:
                evaluation = self.evaluate_question(expected, run)
                answer_evaluation = evaluation.get("answer_evaluation", {})
                tool_evaluation = evaluation.get("tool_evaluation", {})
                report.append({
                    "question_id": expected.get("id"),
                    "question": question,
                    "expected_answer": expected.get("expected_answer", ""),
                    "agent_answer": run.get("final_result", ""),
                    "overall_score": evaluation.get("overall_score"),
                    "answer_score": answer_evaluation.get("answer_match_score"),
                    "answer_matched_count": answer_evaluation.get("answer_matched_count"),
                    "answer_total_count": answer_evaluation.get("answer_total_count"),
                    "tool_f1": tool_evaluation.get("f1_score"),
                    "source_file": run.get("source_file"),
                    "tools": self._join_tool_traces(
                        run.get("tool_calls", []),
                        run.get("tool_results", []),
                    ),
                })

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
    def print_evaluation_report(self, evaluation_results: Dict[str, Any]) -> None:
        if not evaluation_results:
            print("No evaluation results to display")
            return
            
        summary = evaluation_results.get('summary', {})
        aggregated_stats = evaluation_results.get('aggregated_stats', {})
        detailed_evaluations = evaluation_results.get('detailed_evaluations', {})
        
        print("\n" + "="*80)
        print("GDS AGENT BENCHMARK EVALUATION REPORT (MULTIPLE RUNS)")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"Dataset: {summary.get('dataset', 'unknown')}")
        print(f"Model: {summary.get('model', 'unknown')}")
        print(f"Questions File: {summary.get('questions_file', 'unknown')}")
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Total Runs Processed: {summary.get('total_runs', 0)}")
        print(f"Results Files Processed: {summary.get('results_files_processed', 0)}")
        print(f"Average Score Across Questions: {summary.get('average_score_across_questions', 0.0):.2%}")
        
        print(f"\n AGGREGATED RESULTS BY QUESTION:")
        
        for i, (question, stats) in enumerate(aggregated_stats.items(), 1):
            if stats['num_runs'] == 0:
                print(f"\n{i}. ❌ {question[:60]}...")
                print(f"   Error: No runs found")
                continue
                
            mean_score = stats['overall_score_mean']
            success_rate = stats['success_rate']
            
            print(f"\n{i}. Question: {question}")
            print(f"   📊 Runs: {stats['num_runs']}")
            print(f"   📈 Mean Score: {mean_score:.2%} (min: {stats['overall_score_min']:.2%}, max: {stats['overall_score_max']:.2%})")
            print(f"   ✅ Success Rate (>80%): {success_rate:.2%}")
            print(f"   🔧 Tool F1 Mean: {stats['tool_f1_mean']:.2%}")
            print(f"   💬 Answer Score Mean: {stats['answer_score_mean']:.2%}")
            
            evaluation = detailed_evaluations.get(question, {})
            runs = evaluation.get('runs', [])
            if runs:
                print(f"   🏃 Individual Runs:")
                for j, run in enumerate(runs[:3], 1):  # Show first 3 runs
                    print(f"      Run {j}: {run['overall_score']:.2%} ({run.get('source_file', 'unknown')[-20:]})")
                if len(runs) > 3:
                    print(f"      ... and {len(runs) - 3} more runs")
            
        print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GDS Agent Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate_benchmark.py ln                            # Evaluate LN questions with default model
  python evaluate_benchmark.py got                           # Evaluate GoT questions with default model
  python evaluate_benchmark.py ln --model haiku-3-20241022   # Evaluate LN questions for Haiku model
  python evaluate_benchmark.py citations --questions-file questions/gds-questions-citations.json --model sonnet-4-5
  python evaluate_benchmark.py citations --questions-file questions/gds-questions-citations.json --model gpt-5 --question-trace-report"""
    )
    
    parser.add_argument(
        "dataset",
        help="Dataset label: 'ln' or 'got' (uses default questions file), or any custom label (requires --questions-file)."
    )

    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Path to a custom questions file (CSV or JSON). Required for datasets other than ln/got."
    )
    
    parser.add_argument(
        "--model", "-m",
        default="sonnet-4-20250514",
        help="Model to evaluate (default: sonnet-4-20250514). Examples: sonnet-4-20250514, haiku-3-20241022"
    )

    parser.add_argument(
        "--question-trace-report",
        action="store_true",
        help="Also write a per-question JSON of expected vs agent answers, scores, and tool traces (id, name, parameters, result)."
    )
    
    args = parser.parse_args()
    
    print("GDS Agent Benchmark Evaluation Tool")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    try:
        evaluator = BenchmarkEvaluator(
            dataset=args.dataset,
            model=args.model,
            questions_file=args.questions_file,
        )
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Check if questions file exists
    if not evaluator.questions_file.exists():
        print(f"❌ Questions file not found: {evaluator.questions_file}")
        return 1
    
    print(f"Questions file: {evaluator.questions_file}")
    print(f"Results files: {evaluator.results_files}")
    print(f"Evaluation output: {evaluator.evaluation_file}")
    
    # Check if results files exist
    if not evaluator.results_files:
        print(f"❌ No results files found matching pattern: results_{args.model}/{args.dataset}_results_*.json")
        return 1
    
    print("Setup looks good!")
    print("")
    
    try:
        results = evaluator.run_evaluation()
        evaluator.print_evaluation_report(results)
        
        # Save detailed results to the auto-generated file
        evaluator.evaluation_file.parent.mkdir(parents=True, exist_ok=True)
        with open(evaluator.evaluation_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Detailed results saved to: {evaluator.evaluation_file}")

        if args.question_trace_report:
            evaluator.write_question_trace_report(evaluator.question_trace_report_file)
            print(f"✅ Question trace report saved to: {evaluator.question_trace_report_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
