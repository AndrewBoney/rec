#!/usr/bin/env python
"""View benchmark results from test runs."""
import json
from pathlib import Path
from typing import List, Dict


def load_benchmark_results(results_file: Path) -> List[Dict]:
    """Load benchmark results from JSON file."""
    if not results_file.exists():
        print(f"No benchmark results found at {results_file}")
        return []

    with open(results_file) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.4f}s"


def display_results(results: List[Dict]):
    """Display benchmark results in a table format."""
    if not results:
        print("No results to display")
        return

    # Group by test type
    grouped = {}
    for result in results:
        test_name = result["test"]
        if test_name not in grouped:
            grouped[test_name] = []
        grouped[test_name].append(result)

    # Display each test group
    for test_name, test_results in grouped.items():
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print('='*80)

        for i, result in enumerate(test_results, 1):
            print(f"\nRun {i}:")
            for key, value in result.items():
                if key == "test":
                    continue

                # Format based on key name
                if "time" in key.lower() and "seconds" in key.lower():
                    print(f"  {key}: {format_time(value)}")
                elif "time" in key.lower() and "_ms" in key.lower():
                    print(f"  {key}: {value:.2f}ms")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def compare_recent_runs(results: List[Dict], n_runs: int = 2):
    """Compare the most recent N runs for each test."""
    print(f"\n{'='*80}")
    print(f"Comparing Last {n_runs} Runs")
    print('='*80)

    # Group by test type
    grouped = {}
    for result in results:
        test_name = result["test"]
        if test_name not in grouped:
            grouped[test_name] = []
        grouped[test_name].append(result)

    for test_name, test_results in grouped.items():
        if len(test_results) < 2:
            continue

        recent = test_results[-n_runs:]
        print(f"\n{test_name}:")

        # Find time metrics
        time_keys = [k for k in recent[0].keys() if "time" in k.lower() and k != "test"]

        for key in time_keys:
            values = [r.get(key) for r in recent if key in r]
            if len(values) >= 2:
                latest = values[-1]
                previous = values[-2]
                diff_pct = ((latest - previous) / previous) * 100 if previous > 0 else 0

                symbol = "📈" if diff_pct > 5 else "📉" if diff_pct < -5 else "➡️"

                if "seconds" in key.lower():
                    print(f"  {key}: {format_time(latest)} ({symbol} {diff_pct:+.1f}%)")
                else:
                    print(f"  {key}: {latest:.2f} ({symbol} {diff_pct:+.1f}%)")


def main():
    """Main entry point."""
    # Look for benchmark results
    results_file = Path("benchmark_results.json")

    results = load_benchmark_results(results_file)

    if not results:
        print("No benchmark results found.")
        print(f"Run tests with: pytest -m benchmark")
        return

    print(f"\nFound {len(results)} benchmark result(s)")

    # Display all results
    display_results(results)

    # Compare recent runs if we have multiple
    if len(results) >= 2:
        compare_recent_runs(results)


if __name__ == "__main__":
    main()
