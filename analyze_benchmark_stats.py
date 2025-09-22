import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path


def load_evaluation_files(pattern="results/*_evaluation_aggregated.json"):
    """Load all aggregated evaluation JSON files from results folder"""
    files = glob.glob(pattern)
    evaluations = {}

    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                dataset_name = Path(file).stem.replace('_evaluation_aggregated', '').replace('_evaluation', '')
                evaluations[dataset_name] = data
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return evaluations


def extract_metrics(evaluations):
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
    run_variation_data = {}
    
    for dataset_name, data in evaluations.items():
        run_variation_data[dataset_name] = []
        
        # Extract data from detailed_evaluations (individual runs)
        for question, evaluation in data['detailed_evaluations'].items():
            runs = evaluation.get('runs', [])
            
            for run_idx, run_eval in enumerate(runs):
                results['datasets'].append(dataset_name)
                results['overall_scores'].append(run_eval['overall_score'])
                
                tool_eval = run_eval.get('tool_evaluation', {})
                results['tool_precision'].append(tool_eval.get('precision', 0))
                results['tool_recall'].append(tool_eval.get('recall', 0))
                results['tool_f1'].append(tool_eval.get('f1_score', 0))
                results['call_efficiency'].append(tool_eval.get('call_efficiency', 1.0))
                
                param_eval = run_eval.get('parameter_evaluation', {})
                param_scores = [v.get('score', 0) for v in param_eval.values() if isinstance(v, dict)]
                avg_param_score = np.mean(param_scores) if param_scores else 0
                results['parameter_scores'].append(avg_param_score)
                
                answer_eval = run_eval.get('answer_evaluation', {})
                answer_match_score = answer_eval.get('answer_match_score', 0.0)
                if isinstance(answer_match_score, bool):
                    answer_match_score = float(answer_match_score)
                results['answer_scores'].append(answer_match_score)
                results['answer_match_score'].append(answer_match_score)
                
                metadata = run_eval.get('metadata', {})
                results['num_turns'].append(metadata.get('num_turns', 0))
                results['duration_ms'].append(metadata.get('duration_ms', 0))
                
                token_usage = run_eval.get('token_usage', {})
                results['total_input_tokens'].append(token_usage.get('total_input_tokens', 0))
                results['total_output_tokens'].append(token_usage.get('total_output_tokens', 0))
                results['total_cache_creation_tokens'].append(token_usage.get('total_cache_creation_tokens', 0))
                results['total_cache_read_tokens'].append(token_usage.get('total_cache_read_tokens', 0))
                results['total_tokens'].append(token_usage.get('total_tokens', 0))
                results['total_cost_usd'].append(token_usage.get('total_cost_usd', 0.0))
                results['message_count'].append(token_usage.get('message_count', 0))
                
                question_results.append({
                    'dataset': dataset_name,
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'run_number': run_idx + 1,
                    'overall_score': run_eval['overall_score'],
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
                    'source_file': run_eval.get('source_file', ''),
                    'cost_per_question': token_usage.get('total_cost_usd', 0.0)
                })
            
            if runs:
                run_variation_data[dataset_name].append({
                    'question': question,
                    'scores': [run['overall_score'] for run in runs],
                    'tool_f1s': [run.get('tool_evaluation', {}).get('f1_score', 0) for run in runs],
                    'answer_scores': [run.get('answer_evaluation', {}).get('answer_match_score', 0) for run in runs],
                    'num_runs': len(runs)
                })
    
    return results, question_results, run_variation_data


def calculate_summary_stats(results):
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


def create_variation_plots(run_variation_data, dataset_name):
    if not run_variation_data:
        return
        
    plots_dir = Path(f"results/plots_{dataset_name}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Score variation across questions
    plt.figure(figsize=(14, 8))
    
    questions = [item['question'][:30] + '...' if len(item['question']) > 30 else item['question'] 
                for item in run_variation_data]
    
    score_data = [item['scores'] for item in run_variation_data]
    bp = plt.boxplot(score_data, patch_artist=True, labels=range(1, len(questions)+1))
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    plt.xlabel('Question Number')
    plt.ylabel('Overall Score')
    plt.title(f'Score Variation Across Runs - {dataset_name}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'score_variation_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Success rate heatmap
    plt.figure(figsize=(12, 8))
    
    success_rates = []
    question_labels = []
    
    for item in run_variation_data:
        if item['scores']:
            success_rate = sum(1 for score in item['scores'] if score >= 0.8) / len(item['scores'])
            success_rates.append(success_rate)
            question_labels.append(item['question'][:40] + '...' if len(item['question']) > 40 else item['question'])
    
    if success_rates:
        y_pos = np.arange(len(question_labels))
        colors = ['red' if rate < 0.5 else 'yellow' if rate < 0.8 else 'green' for rate in success_rates]
        
        plt.barh(y_pos, success_rates, color=colors, alpha=0.7)
        plt.yticks(y_pos, question_labels)
        plt.xlabel('Success Rate (Score ≥ 0.8)')
        plt.title(f'Success Rate by Question - {dataset_name}')
        plt.xlim(0, 1)
        
        for i, rate in enumerate(success_rates):
            plt.text(rate + 0.01, i, f'{rate:.2f}', va='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_rate_by_question.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Coefficient of variation (CV) analysis
    plt.figure(figsize=(12, 6))
    
    cvs = []
    means = []
    question_nums = []
    
    for i, item in enumerate(run_variation_data):
        if len(item['scores']) > 1:
            mean_score = np.mean(item['scores'])
            std_score = np.std(item['scores'])
            cv = std_score / mean_score if mean_score > 0 else 0
            
            cvs.append(cv)
            means.append(mean_score)
            question_nums.append(i + 1)
    
    if cvs:
        plt.scatter(means, cvs, alpha=0.7, s=60)
        plt.xlabel('Mean Score')
        plt.ylabel('Coefficient of Variation')
        plt.title(f'Score Variability vs Performance - {dataset_name}')
        
        # Add question number annotations
        for i, (mean, cv, qnum) in enumerate(zip(means, cvs, question_nums)):
            plt.annotate(str(qnum), (mean, cv), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'variability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Variation plots for {dataset_name} saved to {plots_dir}/")


def create_visualizations(results, question_results, stats, evaluations, run_variation_data):
    """Create comprehensive visualizations"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 13,
        'figure.titlesize': 18,
        'font.family': 'DejaVu Sans'
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
    
    means = [stats[metric]['mean'] for metric in tool_metrics]
    mins = [stats[metric]['min'] for metric in tool_metrics]
    maxs = [stats[metric]['max'] for metric in tool_metrics]
    
    for i, (color, x, mean_val, min_val, max_val) in enumerate(zip(colors, x_pos, means, mins, maxs)):
        plt.plot([x, x], [min_val, max_val], '-', color=color, linewidth=2, alpha=0.7)
        
        line_width = 0.15
        plt.plot([x - line_width, x + line_width], [min_val, min_val], '-', color=color, linewidth=2, alpha=1.0)
        plt.plot([x - line_width, x + line_width], [mean_val, mean_val], '-', color=color, linewidth=2, alpha=1.0)
        plt.plot([x - line_width, x + line_width], [max_val, max_val], '-', color=color, linewidth=2, alpha=1.0)
        
        label_fontsize = 10
        label_offset = 0.18
        
        # Special handling for Recall (index 1) to avoid overlap between mean and max
        if i == 1 and abs(mean_val - max_val) < 0.05:  # If mean and max are too close in Recall
            mean_offset = -0.02  # Move mean label up slightly
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
    
    min_turns = min(results['num_turns'])
    max_turns = max(results['num_turns'])
    bins = np.arange(min_turns - 0.5, max_turns + 1.5, 1)
    
    plt.hist(results['num_turns'], bins=bins, alpha=0.7, edgecolor='black', color='lightgreen', linewidth=1.2)
    plt.axvline(stats['num_turns']['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["num_turns"]["mean"]:.1f} turns')
    plt.xlabel('Number of Turns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Turns')
    
    plt.xticks(range(min_turns, max_turns + 1))
    
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('turns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 4 saved as 'turns_distribution.png'")
    
    # Chart 5: Token Usage Distribution
    plt.figure(figsize=(10, 6))
    
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
    
    # Create variation plots for each dataset
    print("\nCreating variation analysis plots...")
    for dataset_name, variation_data in run_variation_data.items():
        create_variation_plots(variation_data, dataset_name)


def print_detailed_stats(stats, evaluations):
    print("\n" + "="*80)
    print("BENCHMARK STATISTICS SUMMARY (MULTIPLE RUNS)")
    print("="*80)
    
    total_questions = sum(data['summary']['total_questions'] for data in evaluations.values())
    total_runs = sum(data['summary']['total_runs'] for data in evaluations.values())
    print(f"\nTotal Questions Evaluated: {total_questions}")
    print(f"Total Runs Processed: {total_runs}")
    print(f"Total Datasets: {len(evaluations)}")
    
    print(f"\nDataset Breakdown:")
    for name, data in evaluations.items():
        print(f"  {name}: {data['summary']['total_questions']} questions, "
              f"{data['summary']['total_runs']} runs, "
              f"avg score: {data['summary']['average_score_across_questions']:.3f}")
    
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
                print(f"{metric.replace('_', ' ').title():<20} ${stat['mean']:<11.4f} "
                      f"${stat['std']:<11.4f} ${stat['median']:<11.4f} ${stat['min']:<11.4f} ${stat['max']:<11.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():<20} {stat['mean']:<12.0f} "
                      f"{stat['std']:<12.0f} {stat['median']:<12.0f} {stat['min']:<12.0f} {stat['max']:<12.0f}")
    
    print(f"\nPerformance Thresholds:")
    overall_scores = [run_eval['overall_score'] 
                     for data in evaluations.values()
                     for evaluation in data['detailed_evaluations'].values()
                     for run_eval in evaluation.get('runs', [])]
    thresholds = [0.9, 0.8, 0.7, 0.5]
    for threshold in thresholds:
        success_rate = np.mean([score >= threshold for score in overall_scores]) * 100
        print(f"  Score ≥ {threshold:.1f}: {success_rate:.1f}% ({int(success_rate * len(overall_scores) / 100)}/{len(overall_scores)} questions)")
    
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
    results, question_results, run_variation_data = extract_metrics(evaluations)
    
    print("Calculating summary statistics...")
    stats = calculate_summary_stats(results)
    
    print("Creating visualizations...")
    create_visualizations(results, question_results, stats, evaluations, run_variation_data)
    
    print_detailed_stats(stats, evaluations)
    
    # Save detailed results to CSV
    df = pd.DataFrame(question_results)
    df.to_csv('detailed_question_results.csv', index=False)
    print(f"\nDetailed results saved to 'detailed_question_results.csv'")
    
    print("\nAnalysis complete!")
    print(f"\nVariation analysis plots saved in results/plots_<dataset_name>/ directories")


if __name__ == "__main__":
    main()