
import re

with open('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    row = i + 1
    # Find all div/aside/main/header
    tags = re.findall(r'<(div|aside|main|header)(?:\s+[^>]*?)?>|</(div|aside|main|header)>', line)
    for open_tag, close_tag in tags:
        if open_tag:
            print(f"+ {open_tag} at {row}")
        elif close_tag:
            print(f"- {close_tag} at {row}")
