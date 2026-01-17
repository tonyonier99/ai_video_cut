
import re

with open('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx', 'r') as f:
    content = f.read()

# Very basic tag counter
stack = []
tags = re.findall(r'<([a-zA-Z0-9-]+)|</([a-zA-Z0-9-]+)>', content)

for open_tag, close_tag in tags:
    if open_tag:
        if open_tag not in ['img', 'input', 'br', 'hr', 'link', 'meta']:
            stack.append(open_tag)
    elif close_tag:
        if not stack:
            print(f"Extra closing tag: {close_tag}")
        else:
            last = stack.pop()
            if last != close_tag:
                print(f"Mismatched tag: open={last}, close={close_tag}")

if stack:
    print(f"Unclosed tags: {stack}")
else:
    print("Tags look balanced (basic check)")
