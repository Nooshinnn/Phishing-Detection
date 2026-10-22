"""
inspect_calibration_json.py

Quick diagnostic to print the exact structure of your calibration
results file, so the reliability-diagram loader can be written to match
it correctly instead of guessing.

Usage:
    python inspect_calibration_json.py calibration_full_analysis_results.json
"""
import sys
import json

path = sys.argv[1] if len(sys.argv) > 1 else "calibration_full_analysis_results.json"

with open(path) as f:
    d = json.load(f)


def describe(obj, prefix="", max_depth=4, depth=0):
    if depth > max_depth:
        print(f"{prefix}... (truncated, too deep)")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}: {type(v).__name__}")
                describe(v, prefix + "  ", max_depth, depth + 1)
            else:
                val_preview = str(v)[:60]
                print(f"{prefix}{k}: {type(v).__name__} = {val_preview}")
    elif isinstance(obj, list):
        print(f"{prefix}[list of {len(obj)} items]")
        if len(obj) > 0:
            print(f"{prefix}First item:")
            describe(obj[0], prefix + "  ", max_depth, depth + 1)


print(f"=== Structure of {path} ===\n")
describe(d)
