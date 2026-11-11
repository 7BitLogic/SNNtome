#!/usr/bin/env python3
# ==============================================================================
# scripts/check_duplicates.py
#
# Automated Code Duplication & Structural Clone Detector for C++/HIP codebase.
# Enforces single-source-of-truth and prevents code duplication across suites.
# ==============================================================================

import os
import sys
import re
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Target directories to monitor for duplicates
SOURCE_FILES = [
    "tests/benchmark_whitepaper_k44.cpp",
    "tests/benchmark_batch_scaling.cpp",
    "tests/error_collapse_test.cpp",
    "tests/benchmark_common.cuh",
]

def clean_line(line):
    # Strip comments and extra whitespace
    line = re.sub(r'//.*$', '', line)
    return line.strip()

def is_trivial(line):
    return line in (
        '', '{', '}', '};', 'return;', 'return 0;', 'break;',
        'return true;', 'return false;', '#pragma once',
        '#define CHK(cmd) CHECK_HIP(cmd)'
    ) or line.startswith('#include')

def scan_files(min_lines=4):
    file_lines = {}
    for rel_path in SOURCE_FILES:
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(full_path):
            continue
        with open(full_path, 'r', encoding='utf-8') as f:
            raw = f.readlines()
        cleaned = []
        for i, line in enumerate(raw, 1):
            cleaned.append((i, clean_line(line), line.rstrip()))
        file_lines[rel_path] = cleaned

    # Collect n-grams
    blocks = defaultdict(list)
    for f, lines in file_lines.items():
        for i in range(len(lines) - min_lines + 1):
            window = [lines[i + k][1] for k in range(min_lines)]
            if any(is_trivial(w) for w in window):
                continue
            key = tuple(window)
            blocks[key].append((f, lines[i][0]))

    # Expand maximal duplicate blocks
    duplicates = []
    for window, occurrences in blocks.items():
        if len(occurrences) > 1:
            for i1 in range(len(occurrences)):
                for i2 in range(i1 + 1, len(occurrences)):
                    f1, s1 = occurrences[i1]
                    f2, s2 = occurrences[i2]
                    # Expand forward
                    k = 0
                    while True:
                        l1 = s1 - 1 + k
                        l2 = s2 - 1 + k
                        if l1 >= len(file_lines[f1]) or l2 >= len(file_lines[f2]):
                            break
                        c1 = file_lines[f1][l1][1]
                        c2 = file_lines[f2][l2][1]
                        if c1 == c2 and not is_trivial(c1):
                            k += 1
                        elif c1 == c2 and is_trivial(c1) and k > 0:
                            k += 1
                        else:
                            break
                    if k >= min_lines:
                        duplicates.append((k, f1, s1, s1 + k - 1, f2, s2, s2 + k - 1))

    # Keep maximal spans
    maximal = []
    for item in sorted(duplicates, key=lambda x: -x[0]):
        k, f1, s1, e1, f2, s2, e2 = item
        subsumed = False
        for mk, mf1, ms1, me1, mf2, ms2, me2 in maximal:
            if f1 == mf1 and f2 == mf2 and ms1 <= s1 and e1 <= me1 and ms2 <= s2 and e2 <= me2:
                subsumed = True
                break
        if not subsumed:
            maximal.append(item)

    return file_lines, maximal

def main():
    print("Checking codebase for forbidden code duplications (> 3 identical lines)...")
    file_lines, duplicates = scan_files(min_lines=4)

    # Filter: focus on cross-file duplications or significant (> 4 lines) copy-paste blocks
    violations = []
    for k, f1, s1, e1, f2, s2, e2 in duplicates:
        # Cross-file duplicates are strictly forbidden between test suites
        if f1 != f2:
            violations.append((k, f1, s1, e1, f2, s2, e2))
        elif k >= 8:
            violations.append((k, f1, s1, e1, f2, s2, e2))

    if violations:
        print("\n" + "=" * 80)
        print(" [DUPLICATION VIOLATION] Forbidden copy-pasted code detected!")
        print("=" * 80)
        print(" POLICY:")
        print("   Duplicated harnesses, data generators, or measurement logic are strictly prohibited.")
        print("   Shared logic must be centralized in tests/benchmark_common.cuh.")
        print("=" * 80)
        for k, f1, s1, e1, f2, s2, e2 in violations:
            print(f"\n- {k} identical lines between:")
            print(f"    File A: {f1}:{s1}-{e1}")
            print(f"    File B: {f2}:{s2}-{e2}")
            print("  Sample Code:")
            for i in range(min(4, k)):
                print(f"    | {file_lines[f1][s1 - 1 + i][2]}")
        print("\nACTION REQUIRED: Refactor duplicated blocks into shared utilities.\n")
        sys.exit(1)
    else:
        print("[DUPLICATE CHECK] OK: No forbidden code duplications or clones found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
