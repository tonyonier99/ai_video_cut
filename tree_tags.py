
import re

with open('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx', 'r') as f:
    lines = f.readlines()

depth = 0
for i, line in enumerate(lines):
    row = i + 1
    # Find all <div, <main, <aside, <header
    # Use a simple regex that skips some complex cases but gets the structure
    line_tags = re.findall(r'<(div|aside|main|header)(?:\s+[^>]*?)?>|</(div|aside|main|header)>', line)
    for open_t, close_t in line_tags:
        if open_t:
            print(f"{'  ' * depth}[{row}] <{open_t}>")
            depth += 1
        elif close_t:
            depth -= 1
            print(f"{'  ' * depth}[{row}] </{close_t}>")

print(f"Final Depth: {depth}")
