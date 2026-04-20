import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def read_benchmark(csv_path):
    models = defaultdict(lambda: {'tokens_per_sec': [], 'elapsed_sec': [], 'ram_gb': [], 'load_time_sec': []})
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            models[model]['tokens_per_sec'].append(float(row['tokens_per_sec']))
            models[model]['elapsed_sec'].append(float(row['elapsed_sec']))
            models[model]['ram_gb'].append(float(row['ram_gb']))
            models[model]['load_time_sec'].append(float(row['load_time_sec']))
    return models


def summarize(models):
    summary = {}
    for model, data in models.items():
        summary[model] = {
            'tokens_per_sec': sum(data['tokens_per_sec']) / len(data['tokens_per_sec']) if data['tokens_per_sec'] else 0,
            'elapsed_sec': sum(data['elapsed_sec']) / len(data['elapsed_sec']) if data['elapsed_sec'] else 0,
            'ram_gb': sum(data['ram_gb']) / len(data['ram_gb']) if data['ram_gb'] else 0,
            'load_time_sec': sum(data['load_time_sec']) / len(data['load_time_sec']) if data['load_time_sec'] else 0,
        }
    return summary


def plot_summary(summary, output_path):
    models = list(summary.keys())
    tokens = [summary[m]['tokens_per_sec'] for m in models]
    latency = [summary[m]['elapsed_sec'] for m in models]
    ram = [summary[m]['ram_gb'] for m in models]
    load_time = [summary[m]['load_time_sec'] for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    axes[0].bar(models, tokens, color=['#407bff', '#f27f3a'])
    axes[0].set_title('Avg tokens/sec')
    axes[0].set_ylabel('Tokens per second')
    axes[0].set_ylim(0, max(tokens) * 1.3)

    axes[1].bar(models, latency, color=['#2ca02c', '#d62728'])
    axes[1].set_title('Avg prompt latency')
    axes[1].set_ylabel('Seconds')
    axes[1].set_ylim(0, max(latency) * 1.3)

    axes[2].bar(models, ram, color=['#9467bd', '#8c564b'])
    axes[2].set_title('Avg RAM usage')
    axes[2].set_ylabel('GB')
    axes[2].set_ylim(0, max(ram) * 1.3)

    axes[3].bar(models, load_time, color=['#2ca02c', '#ff7f0e'])
    axes[3].set_title('Avg model load time')
    axes[3].set_ylabel('Seconds')
    axes[3].set_ylim(0, max(load_time) * 1.3)

    for ax in axes:
        ax.set_xlabel('Model')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Qwen Benchmark Summary', fontsize=16)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Qwen benchmark results from CSV.')
    parser.add_argument('--csv', default='qwen_benchmark_results.csv', help='Input benchmark CSV file')
    parser.add_argument('--out', default='qwen_benchmark.png', help='Output image file')
    args = parser.parse_args()

    models = read_benchmark(args.csv)
    if not models:
        raise SystemExit('No benchmark rows found in %s' % args.csv)

    summary = summarize(models)
    plot_summary(summary, args.out)
    print(f'Created benchmark figure: {args.out}')


if __name__ == '__main__':
    main()
