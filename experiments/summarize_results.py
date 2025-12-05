#!/usr/bin/env python
"""
Summarize experimental results from all optimization methods
"""
import os
import re
import json
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file to extract final loss and accuracy"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    results = {
        'train_loss': None,
        'test_loss': None,
        'test_accuracy': None,
        'final_epoch': None
    }

    # Parse last epoch results
    for line in reversed(lines):
        # Look for pattern: "Epoch [100/100], Train Loss: 0.12345678, Test Loss: 0.12345678"
        match = re.search(r'Epoch \[(\d+)/(\d+)\], Train Loss: ([\d.]+), Test Loss: ([\d.]+)', line)
        if match:
            results['final_epoch'] = int(match.group(1))
            results['train_loss'] = float(match.group(3))
            results['test_loss'] = float(match.group(4))

            # Check for accuracy in the same line
            acc_match = re.search(r'Test Accuracy: ([\d.]+)%', line)
            if acc_match:
                results['test_accuracy'] = float(acc_match.group(1))
            break

    return results

def generate_summary():
    """Generate comprehensive results summary"""
    results_dir = Path('results')

    if not results_dir.exists():
        print("No results directory found. Please run experiments first.")
        return

    summary = {
        'regression': {},
        'classification': {}
    }

    methods = ['SAV', 'ESAV', 'IEQ', 'SGD', 'Adam']

    print("=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Regression Results
    print("REGRESSION TASK (Gaussian Data)")
    print("-" * 80)
    print(f"{'Method':<20} {'Train Loss':<15} {'Test Loss':<15} {'Epochs':<10}")
    print("-" * 80)

    for method in methods:
        log_path = results_dir / method / 'regression_log.txt'
        result = parse_log_file(log_path)

        if result and result['train_loss'] is not None:
            summary['regression'][method] = result
            print(f"{method:<20} {result['train_loss']:<15.8f} {result['test_loss']:<15.8f} {result['final_epoch']:<10}")
        else:
            print(f"{method:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

    # IEQ Adaptive
    log_path = results_dir / 'IEQ' / 'regression_adaptive_log.txt'
    result = parse_log_file(log_path)
    if result and result['train_loss'] is not None:
        summary['regression']['IEQ_Adaptive'] = result
        print(f"{'IEQ_Adaptive':<20} {result['train_loss']:<15.8f} {result['test_loss']:<15.8f} {result['final_epoch']:<10}")

    print()
    print()

    # Classification Results
    print("CLASSIFICATION TASK (MNIST)")
    print("-" * 80)
    print(f"{'Method':<20} {'Train Loss':<15} {'Test Loss':<15} {'Accuracy (%)':<15} {'Epochs':<10}")
    print("-" * 80)

    for method in methods:
        log_path = results_dir / method / 'classification_log.txt'
        result = parse_log_file(log_path)

        if result and result['train_loss'] is not None:
            summary['classification'][method] = result
            acc_str = f"{result['test_accuracy']:.2f}" if result['test_accuracy'] else 'N/A'
            print(f"{method:<20} {result['train_loss']:<15.8f} {result['test_loss']:<15.8f} {acc_str:<15} {result['final_epoch']:<10}")
        else:
            print(f"{method:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

    # IEQ Adaptive
    log_path = results_dir / 'IEQ' / 'classification_adaptive_log.txt'
    result = parse_log_file(log_path)
    if result and result['train_loss'] is not None:
        summary['classification']['IEQ_Adaptive'] = result
        acc_str = f"{result['test_accuracy']:.2f}" if result['test_accuracy'] else 'N/A'
        print(f"{'IEQ_Adaptive':<20} {result['train_loss']:<15.8f} {result['test_loss']:<15.8f} {acc_str:<15} {result['final_epoch']:<10}")

    print()
    print("=" * 80)

    # Save summary to JSON
    summary_path = results_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed summary saved to: {summary_path}")

    # Generate comparison insights
    print("\nKEY INSIGHTS:")
    print("-" * 80)

    if summary['regression']:
        print("\nRegression Task:")
        best_method = min(summary['regression'].items(),
                         key=lambda x: x[1]['test_loss'] if x[1]['test_loss'] else float('inf'))
        print(f"  • Best Test Loss: {best_method[0]} ({best_method[1]['test_loss']:.8f})")

    if summary['classification']:
        print("\nClassification Task:")
        best_method = max(summary['classification'].items(),
                         key=lambda x: x[1]['test_accuracy'] if x[1]['test_accuracy'] else 0)
        if best_method[1]['test_accuracy']:
            print(f"  • Best Test Accuracy: {best_method[0]} ({best_method[1]['test_accuracy']:.2f}%)")

    print()

if __name__ == '__main__':
    generate_summary()
