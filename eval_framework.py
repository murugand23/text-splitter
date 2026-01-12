#!/usr/bin/env python3
"""
Minimal evaluation framework for document splitting.

Measures:
- Correctness: Content preservation + exact count
- Quality: Heading accuracy + distribution
- Efficiency: Cost + speed + first-attempt success

Usage:
    python eval_framework.py
"""

import time
import statistics
from typing import List, Dict, Tuple
from hw import split_document, count_tokens
import re


def evaluate_approach(doc: str, target_slides: int, doc_name: str) -> dict:
    """
    Evaluate one document split.
    Returns all metrics needed to assess quality.
    """
    start = time.time()
    slides, meta = split_document(doc, target_slides)
    elapsed = time.time() - start
    
    # Correctness (must be 100%)
    content_preserved = ''.join(slides) == doc
    exact_count = len(slides) == target_slides
    
    # Quality: Heading placement
    orphaned = 0
    total_headings = 0
    for slide in slides:
        lines = [l.strip() for l in slide.strip().split('\n') if l.strip()]
        if not lines:
            continue
        for line in lines:
            if re.match(r'^#{1,6}\s+\w', line):
                total_headings += 1
        if lines and re.match(r'^#{1,6}\s+\w', lines[-1]):
            orphaned += 1
    
    heading_accuracy = ((total_headings - orphaned) / total_headings * 100) if total_headings > 0 else 100
    
    # Quality: Distribution
    token_counts = [count_tokens(s) for s in slides]
    avg_tokens = statistics.mean(token_counts)
    max_avg_ratio = max(token_counts) / avg_tokens if avg_tokens > 0 else 0
    
    return {
        'doc_name': doc_name,
        'target': target_slides,
        'actual': len(slides),
        # Correctness
        'preserved': content_preserved,
        'exact': exact_count,
        'correct': content_preserved and exact_count,
        # Quality
        'heading_accuracy': heading_accuracy,
        'orphaned': orphaned,
        'distribution': max_avg_ratio,
        # Performance
        'cost': meta['cost'],
        'time': elapsed,
        'attempts': meta['attempts'],
        'first_attempt': meta['attempts'] == 1,
        # Raw data
        'token_counts': token_counts,
        'slides': slides
    }


def run_evaluation(test_files: List[Tuple[str, int]]):
    """
    Run evaluation on multiple test files and print results.
    
    Args:
        test_files: List of (filename, target_slides) tuples
    """
    print("="*80)
    print("DOCUMENT SPLITTING EVALUATION")
    print("="*80)
    
    # Load and evaluate each document
    results = []
    for filename, target in test_files:
        try:
            with open(filename, 'r') as f:
                doc = f.read()
            
            print(f"\n{filename} → {target} slides...", end=' ')
            result = evaluate_approach(doc, target, filename)
            results.append(result)
            
            if result['correct']:
                print(f"✅ ({result['time']:.1f}s, ${result['cost']:.4f})")
            else:
                print(f"❌ FAILED")
                
        except FileNotFoundError:
            print(f"\n⚠️  Skipping {filename} (not found)")
    
    if not results:
        print("\n❌ No test files found")
        return
    
    # Print summary
    passing = [r for r in results if r['correct']]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPass Rate: {len(passing)}/{len(results)} ({len(passing)/len(results)*100:.0f}%)")
    
    if passing:
        print(f"\n{'Metric':<25} | {'Value':<15}")
        print("-"*45)
        print(f"{'Content Preservation':<25} | {'✅ 100%':<15}")
        print(f"{'Exact Count':<25} | {'✅ 100%':<15}")
        print(f"{'Heading Accuracy':<25} | {statistics.mean(r['heading_accuracy'] for r in passing):.1f}%")
        print(f"{'Distribution (max/avg)':<25} | {statistics.mean(r['distribution'] for r in passing):.2f}x")
        print(f"{'Average Cost':<25} | ${statistics.mean(r['cost'] for r in passing):.4f}")
        print(f"{'Average Speed':<25} | {statistics.mean(r['time'] for r in passing):.2f}s")
        print(f"{'First-Attempt Success':<25} | {sum(1 for r in passing if r['first_attempt'])/len(passing)*100:.0f}%")
        
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print("="*80)
        
        for r in passing:
            print(f"\n{r['doc_name']}:")
            print(f"  Slides: {r['actual']}/{r['target']}")
            print(f"  Heading accuracy: {r['heading_accuracy']:.1f}% ({r['orphaned']} orphaned)")
            print(f"  Distribution: {r['distribution']:.2f}x")
            print(f"  Cost: ${r['cost']:.4f}")
            print(f"  Time: {r['time']:.2f}s")
            print(f"  Attempts: {r['attempts']}")
            print(f"  Token distribution: min={min(r['token_counts'])}, max={max(r['token_counts'])}, avg={statistics.mean(r['token_counts']):.0f}")
    
    print(f"\n{'='*80}")
    print("✅ Evaluation complete!")
    print("="*80 + "\n")


def main():
    """Run evaluation framework."""
    test_files = [
        ('example.md', 10),
        ('test1.md', 10),
        ('kubernetes.md', 5),
    ]
    
    run_evaluation(test_files)


if __name__ == '__main__':
    main()
