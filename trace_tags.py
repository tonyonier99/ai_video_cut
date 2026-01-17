
import re

with open('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx', 'r') as f:
    lines = f.readlines()

stack = []
for i, line in enumerate(lines):
    row = i + 1
    # Strip comments
    line = re.sub(r'\{/\*.*?\*/\}', '', line)
    # Find all div/aside/main/section/header tags (ignoring self-closing)
    tags = re.findall(r'<(div|aside|main|section|header)(?:\s+[^>]*?)?>|</(div|aside|main|section|header)>', line)
    for open_tag, close_tag in tags:
        if open_tag:
            stack.append((open_tag, row))
        elif close_tag:
            if not stack:
                print(f"[{row}] Extra closing tag: {close_tag}")
            else:
                last_tag, last_row = stack.pop()
                if last_tag != close_tag:
                    print(f"[{row}] Mismatched tag: open={last_tag} (line {last_row}), close={close_tag}")

for tag, row in stack:
    print(f"Unclosed tag: {tag} at line {row}")
