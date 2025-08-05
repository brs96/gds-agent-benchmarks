#!/usr/bin/env python3
"""
GDS Agent Benchmark Evaluation Tool

This script evaluates the performance of the GDS agent by comparing
actual results from benchmark_results.json against expected results
from the enhanced CSV file.
"""

import json
import logging
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Import the fine-grained path questions evaluator functions
from path_questions_evaluation import evaluate_path_algorithm_output

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    def __init__(self, 
                 questions_file: str = "path_questions_basic.csv",
                 results_file: str = None,
                 evaluation_file: str = None):
        self.questions_file = Path(questions_file)
        
        
        # Auto-derive results file from questions file if not provided
        if results_file is None:
            # Convert x.csv to x_results.json
            base_name = self.questions_file.stem
            self.results_file = Path(f"{base_name}_results.json")
        else:
            self.results_file = Path(results_file)
            
        # Auto-derive evaluation file from questions file if not provided
        if evaluation_file is None:
            # Convert x.csv to x_evaluation.json
            base_name = self.questions_file.stem
            self.evaluation_file = Path(f"{base_name}_evaluation.json")
        else:
            self.evaluation_file = Path(evaluation_file)
        
    def load_expected_results(self) -> Dict[str, Dict[str, Any]]:
        """Load expected results from CSV file with new 4-lines-per-question format."""
        expected = {}
        
        if not self.questions_file.exists():
            logger.error(f"Questions file not found: {self.questions_file}")
            return expected
            
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
                
                if not lines:
                    logger.error("Questions file is empty")
                    return expected
                
                # Skip header line if present
                start_idx = 0
                if lines[0].lower().startswith(('question', 'expected_tools', 'expected_parameters', 'expected_answer')):
                    start_idx = 1
                
                # Process lines in groups of 4: question, tools, parameters, answer
                i = start_idx
                while i + 3 < len(lines):
                    # Line i: question (clean format - no quotes needed)
                    question = lines[i].strip()
                    
                    # Line i+1: expected_tools (clean format - no quotes needed)
                    tools_str = lines[i+1].strip()
                    
                    # Line i+2: expected_parameters (clean format - no quotes needed)
                    params_str = lines[i+2].strip()
                    
                    # Line i+3: expected_answer (clean format - no quotes needed)
                    answer = lines[i+3].strip()
                    
                    if question:
                        try:
                            expected[question] = {
                                'expected_tools': json.loads(tools_str),
                                'expected_parameters': json.loads(params_str),
                                'expected_answer': answer
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON for question: {question[:50]}...")
                            logger.warning(f"Tools string: '{tools_str}'")
                            logger.warning(f"Params string: '{params_str}'")
                            logger.warning(f"JSON Error: {e}")
                    
                    i += 4  # Move to next question block
                    
        except Exception as e:
            logger.error(f"Error loading expected results: {e}")
            
        logger.info(f"Loaded {len(expected)} expected results")
        return expected
        
    def load_actual_results(self) -> Dict[str, Dict[str, Any]]:
        """Load actual results from benchmark results JSON."""
        actual = {}
        
        if not self.results_file.exists():
            logger.error(f"Results file not found: {self.results_file}")
            return actual
            
        try:
            with open(self.results_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            for result in data.get('raw_results', []):
                if result.get('success', False) and result.get('response_data'):
                    question = result['question'].strip()
                    response_data = result['response_data']
                    
                    actual[question] = {
                        'tool_calls': response_data.get('tool_calls', []),
                        'tool_results': response_data.get('tool_results', []),
                        'final_result': response_data.get('final_result', ''),
                        'num_turns': response_data.get('num_turns', 0),
                        'duration_ms': response_data.get('duration_ms', 0)
                    }
                    
        except Exception as e:
            logger.error(f"Error loading actual results: {e}")
            
        logger.info(f"Loaded {len(actual)} actual results")
        return actual
        
    def evaluate_tool_calls(self, expected_tools: List[str], actual_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if the correct tools were called."""
        actual_tool_names = [tool['name'] for tool in actual_tools]
        
        # Check if all expected tools were called
        missing_tools = [tool for tool in expected_tools if tool not in actual_tool_names]
        unexpected_tools = [tool for tool in actual_tool_names if tool not in expected_tools]
        
        # Calculate precision and recall
        if len(expected_tools) == 0:
            precision = 1.0 if len(actual_tool_names) == 0 else 0.0
            recall = 1.0
        else:
            correct_tools = len([tool for tool in actual_tool_names if tool in expected_tools])
            precision = correct_tools / len(actual_tool_names) if len(actual_tool_names) > 0 else 0.0
            recall = correct_tools / len(expected_tools)
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'missing_tools': missing_tools,
            'unexpected_tools': unexpected_tools,
            'exact_match': len(missing_tools) == 0 and len(unexpected_tools) == 0
        }
        
    def evaluate_parameters(self, expected_params: Dict[str, Any], actual_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if the correct parameters were used."""
        parameter_scores = {}
        
        for tool_key, expected_tool_params in expected_params.items():
            # Find corresponding actual tool calls
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
                
            # Check parameters for the first matching tool
            actual_params = matching_tools[0].get('parameters', {})
            
            # Compare key parameters
            matches = []
            mismatches = []
            
            for param_key, expected_value in expected_tool_params.items():
                actual_value = actual_params.get(param_key)
                
                if actual_value == expected_value:
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
                'score': len(matches) / len(expected_tool_params) if expected_tool_params else 1.0
            }
            
        return parameter_scores
        
    def evaluate_answer_similarity(self, expected_answer: str, actual_answer: str, expected_tools: List[str] = None) -> Dict[str, Any]:
        """Evaluate similarity between expected and actual answers."""
        
        # Check if this is a path questions file based on filename
        if "path_questions" in str(self.questions_file) and expected_tools:
            # Use fine-grained evaluation for path questions
            result = evaluate_path_algorithm_output(str(expected_tools), expected_answer, actual_answer)
            
            # Convert to expected format for compatibility
            if result.get('success'):
                return {
                    'exact_path_matches': 1,
                    'total_expected_paths': 1,
                    'path_match_score': 1.0,
                    'exact_match': True,
                    'number_match_score': 1.0,
                    'station_coverage': 1.0,
                    'exact_number_matches': 1,
                    'total_expected_numbers': 1
                }
            else:
                return {
                    'exact_path_matches': 0,
                    'total_expected_paths': 1,
                    'path_match_score': 0.0,
                    'exact_match': False,
                    'number_match_score': 0.0,
                    'station_coverage': 0.0,
                    'exact_number_matches': 0,
                    'total_expected_numbers': 1,
                    'error': result.get('error', 'Evaluation failed')
                }
        else:
            # Use existing centrality evaluation for centrality questions
            return self.evaluate_centrality_answer(expected_answer, actual_answer)
    
    def evaluate_centrality_answer(self, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        """Evaluate centrality-based answers with station: score format."""
        
        # Extract result scores more precisely - look for patterns like "Name: score"
        def extract_result_scores(text):
            # Pattern to match "Name: number" or "Name**: number" etc.
            score_pattern = r'([A-Za-z\s\(\)]+)[:*\s]+(\d*\.?\d+)'
            matches = re.findall(score_pattern, text)
            
            scores = {}
            for name, score in matches:
                # Clean up the name
                name = re.sub(r'[*\-\s]+', ' ', name).strip()
                name = re.sub(r'\s+', ' ', name)
                scores[name.lower()] = score
            
            return scores
            
        # Also extract just the numerical scores for fallback comparison
        def extract_score_numbers(text):
            # Look for decimal numbers that appear to be scores (0.xxx format)
            return re.findall(r'\b0\.\d+\b', text)
            
        expected_scores = extract_result_scores(expected_answer)
        actual_scores = extract_result_scores(actual_answer)
        
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
                        if expected_score == actual_score:
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
        """Evaluate a single question's performance."""
        
        # Evaluate tool calls
        tool_evaluation = self.evaluate_tool_calls(
            expected['expected_tools'], 
            actual['tool_calls']
        )
        
        # Evaluate parameters
        parameter_evaluation = self.evaluate_parameters(
            expected['expected_parameters'],
            actual['tool_calls']
        )
        
        # Evaluate answer similarity
        answer_evaluation = self.evaluate_answer_similarity(
            expected['expected_answer'],
            actual['final_result'],
            expected['expected_tools']
        )
        
        # Calculate overall score
        tool_score = tool_evaluation['f1_score']
        param_score = sum(p['score'] for p in parameter_evaluation.values()) / len(parameter_evaluation) if parameter_evaluation else 0.0
        answer_score = answer_evaluation['number_match_score']  # Focus on exact number matches
        
        overall_score = (tool_score + param_score + answer_score) / 3
        
        return {
            'overall_score': overall_score,
            'tool_evaluation': tool_evaluation,
            'parameter_evaluation': parameter_evaluation,
            'answer_evaluation': answer_evaluation,
            'metadata': {
                'num_turns': actual.get('num_turns', 0),
                'duration_ms': actual.get('duration_ms', 0)
            }
        }
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation."""
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
        
        for question in expected_results:
            if question in actual_results:
                logger.info(f"Evaluating: {question[:50]}...")
                evaluation = self.evaluate_question(
                    expected_results[question],
                    actual_results[question]
                )
                evaluations[question] = evaluation
            else:
                logger.warning(f"No actual result found for question: {question[:50]}...")
                evaluations[question] = {
                    'overall_score': 0.0,
                    'error': 'No actual result found'
                }
        
        # Calculate summary statistics
        valid_scores = [e['overall_score'] for e in evaluations.values() if 'error' not in e]
        
        summary = {
            'total_questions': len(expected_results),
            'evaluated_questions': len(valid_scores),
            'average_score': sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
            'scores': valid_scores
        }
        
        return {
            'summary': summary,
            'detailed_evaluations': evaluations
        }
        
    def print_evaluation_report(self, evaluation_results: Dict[str, Any]) -> None:
        """Print a detailed evaluation report."""
        
        if not evaluation_results:
            print("❌ No evaluation results to display")
            return
            
        summary = evaluation_results.get('summary', {})
        evaluations = evaluation_results.get('detailed_evaluations', {})
        
        print("\n" + "="*80)
        print("GDS AGENT BENCHMARK EVALUATION REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Evaluated Questions: {summary.get('evaluated_questions', 0)}")
        print(f"Average Score: {summary.get('average_score', 0.0):.2%}")
        
        print(f"\n📋 DETAILED RESULTS:")
        
        for i, (question, evaluation) in enumerate(evaluations.items(), 1):
            if 'error' in evaluation:
                print(f"\n{i}. ❌ {question[:60]}...")
                print(f"   Error: {evaluation['error']}")
                continue
                
            score = evaluation['overall_score']
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            
            print(f"\n{i}. {status} Score: {score:.2%}")
            print(f"   Question: {question}")
            
            # Tool evaluation
            tool_eval = evaluation['tool_evaluation']
            print(f"   🔧 Tools: P={tool_eval['precision']:.2f}, R={tool_eval['recall']:.2f}, F1={tool_eval['f1_score']:.2f}")
            if tool_eval['missing_tools']:
                print(f"      Missing: {', '.join(tool_eval['missing_tools'])}")
            if tool_eval['unexpected_tools']:
                print(f"      Unexpected: {', '.join(tool_eval['unexpected_tools'])}")
                
            # Parameter evaluation  
            param_eval = evaluation['parameter_evaluation']
            for tool_key, params in param_eval.items():
                match_status = "✓" if params['match'] else "✗"
                print(f"   ⚙️  {tool_key}: {match_status} (Score: {params.get('score', 0):.2f})")
                if params.get('mismatches'):
                    for mismatch in params['mismatches']:
                        print(f"      {mismatch['param']}: expected {mismatch['expected']}, got {mismatch['actual']}")
                        
            # Answer evaluation
            answer_eval = evaluation['answer_evaluation']
            if "path_questions" in str(self.questions_file):
                # Path-specific reporting
                print(f"   💬 Answer: {answer_eval['exact_path_matches']}/{answer_eval['total_expected_paths']} exact path matches ({answer_eval['path_match_score']:.2%})")
                if answer_eval.get('expected_paths') and answer_eval.get('actual_paths'):
                    print(f"      Expected paths: {list(answer_eval['expected_paths'].keys())}")
                    print(f"      Actual paths: {list(answer_eval['actual_paths'].keys())}")
            else:
                # Centrality-specific reporting
                print(f"   💬 Answer: {answer_eval['exact_number_matches']}/{answer_eval['total_expected_numbers']} exact matches ({answer_eval['number_match_score']:.2%})")
                if answer_eval.get('expected_scores') and answer_eval.get('actual_scores'):
                    print(f"      Expected: {answer_eval['expected_scores']}")
                    print(f"      Actual: {answer_eval['actual_scores']}")
            
            # Metadata
            metadata = evaluation['metadata']
            print(f"   ⏱️  Performance: {metadata['num_turns']} turns, {metadata['duration_ms']}ms")
            
        print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GDS Agent Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate_benchmark.py --questions path_questions_basic.csv"""
    )
    
    parser.add_argument(
        "--questions", "-q",
        default="path_questions_basic.csv",
        help="Path to the questions CSV file (default: path_questions_basic.csv)"
    )
    
    args = parser.parse_args()
    
    print("GDS Agent Benchmark Evaluation Tool")
    print("="*50)
    
    # Check if questions file exists
    if not Path(args.questions).exists():
        print(f"❌ Questions file '{args.questions}' not found")
        return 1
    
    evaluator = BenchmarkEvaluator(
        questions_file=args.questions
    )
    
    print(f"Questions file: {evaluator.questions_file}")
    print(f"Results file: {evaluator.results_file}")
    print(f"Evaluation output: {evaluator.evaluation_file}")
    
    # Check if results file exists
    if not evaluator.results_file.exists():
        print(f"❌ Results file '{evaluator.results_file}' not found")
        return 1
    
    print("✅ Setup looks good!")
    print("")
    
    try:
        results = evaluator.run_evaluation()
        evaluator.print_evaluation_report(results)
        
        # Save detailed results to the auto-generated file
        with open(evaluator.evaluation_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {evaluator.evaluation_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
