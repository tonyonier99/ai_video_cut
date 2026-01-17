
import re

def check_jsx(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'\{/\*.*?\*/\}', '', content)
    content = re.sub(r'//.*', '', content)

    # Find all tags
    # Self-closing tags to ignore
    void_elements = {'img', 'input', 'br', 'hr', 'link', 'meta', 'source', 'wbr'}
    
    # Match <Tag or </Tag
    tags = re.findall(r'<([a-zA-Z0-9\.-]+)|</([a-zA-Z0-9\.-]+)>', content)
    
    stack = []
    for open_tag, close_tag in tags:
        if open_tag:
            if open_tag.lower() in void_elements:
                continue
            # Also ignore some common non-tag things that look like tags (e.g. types in code)
            if open_tag in ['string', 'number', 'boolean', 'any', 'void', 'T', 'K', 'V']:
                continue
            stack.append(open_tag)
        elif close_tag:
            if not stack:
                print(f"Error: Founding closing tag </{close_tag}> but stack is empty")
                continue
            last = stack.pop()
            if last != close_tag:
                print(f"Error: Mismatched tag. Expected </{last}> but found </{close_tag}>")
                # Try to recover by searching back in stack? No, just keep going.
    
    if stack:
        print(f"Error: Unclosed tags: {stack}")
    else:
        print("JSX tags are balanced!")

check_jsx('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx')
