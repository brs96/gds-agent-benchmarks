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
        'num_turns': [],
        'duration_ms': []
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
            answer_score = answer_eval.get('path_match_score', answer_eval.get('exact_match', 0))
            if isinstance(answer_score, bool):
                answer_score = float(answer_score)
            results['answer_scores'].append(answer_score)
            
            # Metadata
            metadata = evaluation.get('metadata', {})
            results['num_turns'].append(metadata.get('num_turns', 0))
            results['duration_ms'].append(metadata.get('duration_ms', 0))
            
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
                'answer_score': answer_score,
                'num_turns': metadata.get('num_turns', 0),
                'duration_ms': metadata.get('duration_ms', 0)
            })
    
    return results, question_results


def calculate_summary_stats(results):
    """Calculate summary statistics"""
    stats = {}
    
    for metric in ['overall_scores', 'tool_precision', 'tool_recall', 'tool_f1', 
                   'call_efficiency', 'parameter_scores', 'answer_scores', 'num_turns', 'duration_ms']:
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
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Overall Performance Distribution
    ax1 = plt.subplot(4, 3, 1)
    plt.hist(results['overall_scores'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(stats['overall_scores']['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats["overall_scores"]["mean"]:.3f}')
    plt.xlabel('Overall Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Overall Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Tool Evaluation Metrics - Custom Mean/Min/Max Plot
    ax2 = plt.subplot(4, 3, 2)
    tool_metrics = ['tool_precision', 'tool_recall', 'tool_f1', 'call_efficiency']
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    
    x_pos = np.arange(len(tool_metrics))
    
    # Extract mean, min, max for each metric
    means = [stats[metric]['mean'] for metric in tool_metrics]
    mins = [stats[metric]['min'] for metric in tool_metrics]
    maxs = [stats[metric]['max'] for metric in tool_metrics]
    
    # Create custom error bars from min to max
    lower_errors = [mean - min_val for mean, min_val in zip(means, mins)]
    upper_errors = [max_val - mean for mean, max_val in zip(means, maxs)]
    
    # Plot mean with custom error bars showing full range
    bars = plt.errorbar(x_pos, means, 
                       yerr=[lower_errors, upper_errors],
                       fmt='o', capsize=8, capthick=2, markersize=8,
                       elinewidth=2, alpha=0.8)
    
    # Color each point and error bar differently
    for i, (color, x, mean_val, min_val, max_val) in enumerate(zip(colors, x_pos, means, mins, maxs)):
        plt.plot(x, mean_val, 'o', color=color, markersize=10, alpha=0.8, markeredgecolor='black', markeredgewidth=1)
        plt.plot([x, x], [min_val, max_val], '-', color=color, linewidth=3, alpha=0.6)
        plt.plot(x, min_val, '_', color=color, markersize=12, markeredgewidth=2)
        plt.plot(x, max_val, '_', color=color, markersize=12, markeredgewidth=2)
        
        # Add value labels
        plt.text(x, mean_val + 0.03, f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(x - 0.15, min_val, f'{min_val:.2f}', ha='center', va='center', fontsize=7, color='darkred')
        plt.text(x + 0.15, max_val, f'{max_val:.2f}', ha='center', va='center', fontsize=7, color='darkgreen')
    
    plt.xlabel('Tool Metrics')
    plt.ylabel('Score')
    plt.title('Tool Evaluation Metrics (Mean, Min, Max)')
    plt.xticks(x_pos, ['Precision', 'Recall', 'F1-Score', 'Call Eff'])
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 3. Performance by Dataset
    ax3 = plt.subplot(4, 3, 3)
    df = pd.DataFrame(question_results)
    dataset_stats = df.groupby('dataset')['overall_score'].agg(['mean', 'std', 'count']).reset_index()
    
    bars = plt.bar(dataset_stats['dataset'], dataset_stats['mean'], 
                   yerr=dataset_stats['std'], capsize=5, alpha=0.7, edgecolor='black')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Overall Score')
    plt.title('Performance by Dataset')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, dataset_stats['count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={count}', ha='center', va='bottom')
    
    # 4. Precision vs Recall Scatter
    ax4 = plt.subplot(4, 3, 4)
    datasets = list(set(results['datasets']))
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        mask = np.array(results['datasets']) == dataset
        plt.scatter(np.array(results['tool_precision'])[mask], 
                   np.array(results['tool_recall'])[mask],
                   c=[colors[i]], label=dataset, alpha=0.7, s=50)
    
    plt.xlabel('Tool Precision')
    plt.ylabel('Tool Recall')
    plt.title('Precision vs Recall by Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Add diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 5. Performance vs Complexity (num_turns)
    ax5 = plt.subplot(4, 3, 5)
    plt.scatter(results['num_turns'], results['overall_scores'], alpha=0.6)
    
    # Add trend line
    z = np.polyfit(results['num_turns'], results['overall_scores'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(results['num_turns']), p(sorted(results['num_turns'])), "r--", alpha=0.8)
    
    plt.xlabel('Number of Turns')
    plt.ylabel('Overall Score')
    plt.title('Performance vs Complexity')
    plt.grid(True, alpha=0.3)
    
    # 6. Duration Analysis
    ax6 = plt.subplot(4, 3, 6)
    duration_seconds = [d/1000 for d in results['duration_ms']]
    plt.hist(duration_seconds, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(duration_seconds), color='red', linestyle='--', 
                label=f'Mean: {np.mean(duration_seconds):.1f}s')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Task Duration Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Success Rate by Score Threshold
    ax7 = plt.subplot(4, 3, 7)
    thresholds = np.arange(0, 1.1, 0.1)
    success_rates = []
    
    for threshold in thresholds:
        success_rate = np.mean([score >= threshold for score in results['overall_scores']])
        success_rates.append(success_rate * 100)
    
    plt.plot(thresholds, success_rates, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Score Threshold')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Score Threshold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 105)
    
    # 8. Box plot of key metrics
    ax8 = plt.subplot(4, 3, 8)
    key_metrics = [results['overall_scores'], results['tool_precision'], 
                   results['tool_recall'], results['tool_f1'], results['call_efficiency']]
    labels = ['Overall', 'Precision', 'Recall', 'F1', 'Call Eff']
    
    plt.boxplot(key_metrics, tick_labels=labels)
    plt.ylabel('Score')
    plt.title('Distribution of Key Metrics')
    plt.grid(True, alpha=0.3)
    
    # 9. Parameter vs Answer Performance
    ax9 = plt.subplot(4, 3, 9)
    plt.scatter(results['parameter_scores'], results['answer_scores'], alpha=0.6)
    
    # Add trend line
    valid_indices = [(i, j) for i, j in zip(results['parameter_scores'], results['answer_scores']) 
                     if not (np.isnan(i) or np.isnan(j))]
    if valid_indices:
        param_vals, answer_vals = zip(*valid_indices)
        z = np.polyfit(param_vals, answer_vals, 1)
        p = np.poly1d(z)
        plt.plot(sorted(param_vals), p(sorted(param_vals)), "r--", alpha=0.8)
    
    plt.xlabel('Parameter Score')
    plt.ylabel('Answer Score')
    plt.title('Parameter vs Answer Performance')
    plt.grid(True, alpha=0.3)
    
    # 10. Top Performing Questions
    ax10 = plt.subplot(4, 3, 10)
    df_sorted = df.sort_values('overall_score', ascending=False)
    top_10 = df_sorted.head(10)
    
    bars = plt.barh(range(len(top_10)), top_10['overall_score'])
    plt.yticks(range(len(top_10)), [q[:30] + "..." for q in top_10['question']])
    plt.xlabel('Overall Score')
    plt.title('Top 10 Performing Questions')
    plt.grid(True, alpha=0.3)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, top_10['overall_score'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center')
    
    # 11. Summary Statistics Table
    ax11 = plt.subplot(4, 3, 11)
    ax11.axis('tight')
    ax11.axis('off')
    
    summary_data = []
    for metric, stat in stats.items():
        if metric in ['overall_scores', 'tool_precision', 'tool_recall', 'tool_f1', 'call_efficiency']:
            summary_data.append([
                metric.replace('_', ' ').title(),
                f"{stat['mean']:.3f}",
                f"{stat['std']:.3f}",
                f"{stat['median']:.3f}",
                f"{stat['min']:.3f}",
                f"{stat['max']:.3f}",
                f"{stat['count']}"
            ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Count'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig('benchmark_statistics.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'benchmark_statistics.png'")
    
    return fig


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
    
    print(f"\nPerformance Thresholds:")
    overall_scores = [score for data in evaluations.values() 
                     for score in data['summary']['scores']]
    thresholds = [0.9, 0.8, 0.7, 0.5]
    for threshold in thresholds:
        success_rate = np.mean([score >= threshold for score in overall_scores]) * 100
        print(f"  Score ≥ {threshold:.1f}: {success_rate:.1f}% ({int(success_rate * len(overall_scores) / 100)}/{len(overall_scores)} questions)")


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
    fig = create_visualizations(results, question_results, stats, evaluations)
    
    print_detailed_stats(stats, evaluations)
    
    # Save detailed results to CSV
    df = pd.DataFrame(question_results)
    df.to_csv('detailed_question_results.csv', index=False)
    print(f"\nDetailed results saved to 'detailed_question_results.csv'")
    
    print("\nAnalysis complete!")
    plt.show()


if __name__ == "__main__":
    main()