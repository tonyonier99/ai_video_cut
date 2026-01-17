
import re

with open('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx', 'r') as f:
    content = f.read()

# Improved tag matcher
# Handles comments and string literals (basic)
content = re.sub(r'\{/\*.*?\*/\}', '', content)
content = re.sub(r'//.*', '', content)

stack = []
# Find all <tag and </tag
# Includes motion.div and motion.aside etc.
all_tags = re.findall(r'<([a-zA-Z0-9\.-]+)|</([a-zA-Z0-9\.-]+)>', content)

for open_t, close_t in all_tags:
    if open_t:
        if open_t not in ['img', 'input', 'br', 'hr', 'link', 'meta', 'source', 'wbr']:
            stack.append(open_t)
    else:
        if not stack:
            print(f"Extra close tag: {close_t}")
        else:
            last = stack.pop()
            if last != close_t:
                print(f"Mismatched tag: open={last}, close={close_t}")

if stack:
    print(f"Unclosed tags: {stack}")
else:
    print("Tags are balanced")
