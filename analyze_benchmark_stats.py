import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path


def load_evaluation_files(pattern="*_evaluation.json"):
    """Load all evaluation JSON files matching the pattern"""
    files = glob.glob(pattern)
    evaluations = {}
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Extract dataset name from filename
                dataset_name = Path(file).stem.replace('_evaluation', '')
                evaluations[dataset_name] = data
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return evaluations


def extract_metrics(evaluations):
    """Extract metrics from evaluation data"""
    results = {
        'datasets': [],
        'overall_scores': [],
        'tool_precision': [],
        'tool_recall': [],
        'tool_f1': [],
        'call_efficiency': [],
        'parameter_scores': [],
        'answer_scores': [],
        'answer_match_score': [],
        'num_turns': [],
        'duration_ms': [],
        'total_input_tokens': [],
        'total_output_tokens': [],
        'total_cache_creation_tokens': [],
        'total_cache_read_tokens': [],
        'total_tokens': [],
        'total_cost_usd': [],
        'message_count': []
    }
    
    question_results = []
    
    for dataset_name, data in evaluations.items():
        for question, evaluation in data['detailed_evaluations'].items():
            # Basic info
            results['datasets'].append(dataset_name)
            results['overall_scores'].append(evaluation['overall_score'])
            
            # Tool evaluation metrics
            tool_eval = evaluation.get('tool_evaluation', {})
            results['tool_precision'].append(tool_eval.get('precision', 0))
            results['tool_recall'].append(tool_eval.get('recall', 0))
            results['tool_f1'].append(tool_eval.get('f1_score', 0))
            results['call_efficiency'].append(tool_eval.get('call_efficiency', 1.0))
            
            # Parameter evaluation
            param_eval = evaluation.get('parameter_evaluation', {})
            param_scores = [v.get('score', 0) for v in param_eval.values() if isinstance(v, dict)]
            avg_param_score = np.mean(param_scores) if param_scores else 0
            results['parameter_scores'].append(avg_param_score)
            
            # Answer evaluation
            answer_eval = evaluation.get('answer_evaluation', {})
            answer_match_score = answer_eval.get('answer_match_score', 0.0)
            if isinstance(answer_match_score, bool):
                answer_match_score = float(answer_match_score)
            results['answer_scores'].append(answer_match_score)
            results['answer_match_score'].append(answer_match_score)
            
            # Metadata
            metadata = evaluation.get('metadata', {})
            results['num_turns'].append(metadata.get('num_turns', 0))
            results['duration_ms'].append(metadata.get('duration_ms', 0))
            
            # Token usage
            token_usage = evaluation.get('token_usage', {})
            results['total_input_tokens'].append(token_usage.get('total_input_tokens', 0))
            results['total_output_tokens'].append(token_usage.get('total_output_tokens', 0))
            results['total_cache_creation_tokens'].append(token_usage.get('total_cache_creation_tokens', 0))
            results['total_cache_read_tokens'].append(token_usage.get('total_cache_read_tokens', 0))
            results['total_tokens'].append(token_usage.get('total_tokens', 0))
            results['total_cost_usd'].append(token_usage.get('total_cost_usd', 0.0))
            results['message_count'].append(token_usage.get('message_count', 0))
            
            # Store individual question results
            question_results.append({
                'dataset': dataset_name,
                'question': question[:100] + "..." if len(question) > 100 else question,
                'overall_score': evaluation['overall_score'],
                'tool_precision': tool_eval.get('precision', 0),
                'tool_recall': tool_eval.get('recall', 0),
                'tool_f1': tool_eval.get('f1_score', 0),
                'call_efficiency': tool_eval.get('call_efficiency', 1.0),
                'param_score': avg_param_score,
                'answer_score': answer_match_score,
                'answer_match_score': answer_match_score,
                'num_turns': metadata.get('num_turns', 0),
                'duration_ms': metadata.get('duration_ms', 0),
                'total_input_tokens': token_usage.get('total_input_tokens', 0),
                'total_output_tokens': token_usage.get('total_output_tokens', 0),
                'total_tokens': token_usage.get('total_tokens', 0),
                'total_cost_usd': token_usage.get('total_cost_usd', 0.0),
                'cost_per_question': token_usage.get('total_cost_usd', 0.0)
            })
    
    return results, question_results


def calculate_summary_stats(results):
    """Calculate summary statistics"""
    stats = {}
    
    for metric in ['overall_scores', 'tool_precision', 'tool_recall', 'tool_f1', 
                   'call_efficiency', 'parameter_scores', 'answer_scores', 'answer_match_score',
                   'num_turns', 'duration_ms', 'total_input_tokens', 'total_output_tokens', 'total_tokens', 'total_cost_usd']:
        values = results[metric]
        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return stats


def create_visualizations(results, question_results, stats, evaluations):
    """Create comprehensive visualizations"""
    
    # Set up the plotting style and fonts
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,           # Default font size
        'axes.titlesize': 16,      # Title font size (larger)
        'axes.labelsize': 14,      # Axis labels font size (larger)
        'xtick.labelsize': 11,     # X-axis tick labels
        'ytick.labelsize': 11,     # Y-axis tick labels
        'legend.fontsize': 13,     # Legend font size (larger)
        'figure.titlesize': 18,    # Figure title font size (larger)
        'font.family': 'DejaVu Sans'  # Better font family
    })
    
    # Chart 1: F1 Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['tool_f1'], bins=20, alpha=0.7, edgecolor='black', color='orange', linewidth=1.2)
    plt.axvline(stats['tool_f1']['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["tool_f1"]["mean"]:.3f}')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of F1 Scores')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 1 saved as 'f1_score_distribution.png'")
    
    # Chart 2: Tool Evaluation Metrics - Custom Mean/Min/Max Plot
    plt.figure(figsize=(12, 6))
    tool_metrics = ['tool_precision', 'tool_recall', 'tool_f1', 'call_efficiency']
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    
    x_pos = np.arange(len(tool_metrics))
    
    # Extract mean, min, max for each metric
    means = [stats[metric]['mean'] for metric in tool_metrics]
    mins = [stats[metric]['min'] for metric in tool_metrics]
    maxs = [stats[metric]['max'] for metric in tool_metrics]
    
    # Draw clean lines for min, mean, max for each metric
    for i, (color, x, mean_val, min_val, max_val) in enumerate(zip(colors, x_pos, means, mins, maxs)):
        # Vertical line from min to max
        plt.plot([x, x], [min_val, max_val], '-', color=color, linewidth=2, alpha=0.7)
        
        # Smaller, cleaner horizontal lines for min, mean, max - matching vertical line color
        line_width = 0.15  # Smaller width for horizontal lines
        plt.plot([x - line_width, x + line_width], [min_val, min_val], '-', color=color, linewidth=2, alpha=1.0)
        plt.plot([x - line_width, x + line_width], [mean_val, mean_val], '-', color=color, linewidth=2, alpha=1.0)
        plt.plot([x - line_width, x + line_width], [max_val, max_val], '-', color=color, linewidth=2, alpha=1.0)
        
        # Add consistent value labels with same font size - all on the right, closer to lines
        label_fontsize = 10
        label_offset = 0.18  # Closer to the lines
        
        # Special handling for Recall (index 1) to avoid overlap between mean and max
        if i == 1 and abs(mean_val - max_val) < 0.05:  # If mean and max are too close in Recall
            mean_offset = -0.02  # Move mean label up slightly, but not too much
        else:
            mean_offset = 0
        
        plt.text(x + label_offset, mean_val + mean_offset, f'{mean_val:.3f}', ha='left', va='center', fontsize=label_fontsize, color='black')
        plt.text(x + label_offset, min_val, f'{min_val:.2f}', ha='left', va='center', fontsize=label_fontsize, color='black')
        plt.text(x + label_offset, max_val, f'{max_val:.2f}', ha='left', va='center', fontsize=label_fontsize, color='black')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Benchmark Evaluation Metrics (Mean, Min, Max)')
    plt.xticks(x_pos, ['Precision', 'Recall', 'F1-Score', 'Call Eff'])
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tool_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 2 saved as 'tool_metrics.png'")
    
    # Chart 3: Duration Analysis
    plt.figure(figsize=(10, 6))
    duration_seconds = [d/1000 for d in results['duration_ms']]
    plt.hist(duration_seconds, bins=20, alpha=0.7, edgecolor='black', color='lightblue', linewidth=1.2)
    plt.axvline(np.mean(duration_seconds), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(duration_seconds):.1f}s')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Task Duration Distribution')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 3 saved as 'duration_distribution.png'")
    
    # Chart 4: Number of Turns Distribution
    plt.figure(figsize=(10, 6))
    
    # Create bins aligned with integer values
    min_turns = min(results['num_turns'])
    max_turns = max(results['num_turns'])
    bins = np.arange(min_turns - 0.5, max_turns + 1.5, 1)  # Bins centered on integers
    
    plt.hist(results['num_turns'], bins=bins, alpha=0.7, edgecolor='black', color='lightgreen', linewidth=1.2)
    plt.axvline(stats['num_turns']['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["num_turns"]["mean"]:.1f} turns')
    plt.xlabel('Number of Turns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Turns')
    
    # Set x-axis ticks to integer values
    plt.xticks(range(min_turns, max_turns + 1))
    
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('turns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 4 saved as 'turns_distribution.png'")
    
    # Chart 5: Token Usage Distribution
    plt.figure(figsize=(10, 6))
    
    # Create properly aligned bins for token distribution
    min_tokens = min(results['total_tokens'])
    max_tokens = max(results['total_tokens'])
    bin_width = (max_tokens - min_tokens) / 20
    bins = np.arange(min_tokens, max_tokens + bin_width, bin_width)
    
    plt.hist(results['total_tokens'], bins=bins, alpha=0.7, edgecolor='black', color='lightblue', linewidth=1.2, align='mid')
    plt.axvline(stats['total_tokens']['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["total_tokens"]["mean"]:.0f}')
    plt.xlabel('Total Tokens')
    plt.ylabel('Frequency')
    plt.title('Total Token Usage Distribution')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('token_usage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 5 saved as 'token_usage_analysis.png'")
    
    print("All charts saved successfully!")


def print_detailed_stats(stats, evaluations):
    """Print detailed statistics to console"""
    print("\n" + "="*80)
    print("BENCHMARK STATISTICS SUMMARY")
    print("="*80)
    
    total_questions = sum(data['summary']['total_questions'] for data in evaluations.values())
    print(f"\nTotal Questions Evaluated: {total_questions}")
    print(f"Total Datasets: {len(evaluations)}")
    
    print(f"\nDataset Breakdown:")
    for name, data in evaluations.items():
        print(f"  {name}: {data['summary']['total_questions']} questions, "
              f"avg score: {data['summary']['average_score']:.3f}")
    
    print(f"\n{'Metric':<20} {'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 68)
    
    for metric, stat in stats.items():
        if metric in ['overall_scores', 'tool_precision', 'tool_recall', 'tool_f1', 
                      'call_efficiency', 'parameter_scores', 'answer_scores']:
            print(f"{metric.replace('_', ' ').title():<20} {stat['mean']:<8.3f} "
                  f"{stat['std']:<8.3f} {stat['median']:<8.3f} {stat['min']:<8.3f} {stat['max']:<8.3f}")
    
    print(f"\n{'Answer Evaluation Metric':<20} {'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 68)
    
    # Single answer evaluation metric
    if 'answer_match_score' in stats:
        stat = stats['answer_match_score']
        print(f"{'Answer Match Score':<20} {stat['mean']:<8.3f} "
              f"{stat['std']:<8.3f} {stat['median']:<8.3f} {stat['min']:<8.3f} {stat['max']:<8.3f}")
    
    print(f"\n{'Token & Cost Metrics':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-" * 88)
    
    # Token metrics
    for metric in ['total_tokens', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd']:
        if metric in stats:
            stat = stats[metric]
            if metric == 'total_cost_usd':
                # Format cost with more precision
                print(f"{metric.replace('_', ' ').title():<20} ${stat['mean']:<11.4f} "
                      f"${stat['std']:<11.4f} ${stat['median']:<11.4f} ${stat['min']:<11.4f} ${stat['max']:<11.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():<20} {stat['mean']:<12.0f} "
                      f"{stat['std']:<12.0f} {stat['median']:<12.0f} {stat['min']:<12.0f} {stat['max']:<12.0f}")
    
    print(f"\nPerformance Thresholds:")
    overall_scores = [score for data in evaluations.values() 
                     for score in data['summary']['scores']]
    thresholds = [0.9, 0.8, 0.7, 0.5]
    for threshold in thresholds:
        success_rate = np.mean([score >= threshold for score in overall_scores]) * 100
        print(f"  Score ≥ {threshold:.1f}: {success_rate:.1f}% ({int(success_rate * len(overall_scores) / 100)}/{len(overall_scores)} questions)")
    
    # Cost analysis summary
    if 'total_cost_usd' in stats:
        total_benchmark_cost = sum(cost for data in evaluations.values() 
                                 for eval_data in data['detailed_evaluations'].values() 
                                 for cost in [eval_data.get('token_usage', {}).get('total_cost_usd', 0.0)])
        
        print(f"\n{'Cost Analysis Summary'}")
        print("-" * 30)
        print(f"Total Benchmark Cost: ${total_benchmark_cost:.4f}")
        print(f"Average Cost per Question: ${stats['total_cost_usd']['mean']:.4f}")
        print(f"Most Expensive Question: ${stats['total_cost_usd']['max']:.4f}")
        print(f"Least Expensive Question: ${stats['total_cost_usd']['min']:.4f}")
        
        # Estimated monthly costs if run daily
        daily_cost = total_benchmark_cost
        monthly_cost = daily_cost * 30
        print(f"Estimated Monthly Cost (if run daily): ${monthly_cost:.2f}")
        
        # Token efficiency
        if 'total_tokens' in stats:
            avg_cost_per_1k_tokens = (stats['total_cost_usd']['mean'] / stats['total_tokens']['mean']) * 1000 if stats['total_tokens']['mean'] > 0 else 0
            print(f"Average Cost per 1K Tokens: ${avg_cost_per_1k_tokens:.4f}")


def main():
    """Main execution function"""
    print("Loading evaluation files...")
    evaluations = load_evaluation_files()
    
    if not evaluations:
        print("No evaluation files found matching '*_evaluation.json' pattern")
        return
    
    print(f"Found {len(evaluations)} evaluation files:")
    for name in evaluations.keys():
        print(f"  - {name}")
    
    print("\nExtracting metrics...")
    results, question_results = extract_metrics(evaluations)
    
    print("Calculating summary statistics...")
    stats = calculate_summary_stats(results)
    
    print("Creating visualizations...")
    create_visualizations(results, question_results, stats, evaluations)
    
    print_detailed_stats(stats, evaluations)
    
    # Save detailed results to CSV
    df = pd.DataFrame(question_results)
    df.to_csv('detailed_question_results.csv', index=False)
    print(f"\nDetailed results saved to 'detailed_question_results.csv'")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()