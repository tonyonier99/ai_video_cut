
import re

def check_structure(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    stack = []
    # Match tags: ignore self-closing with /> and void elements
    void_elements = {'img', 'input', 'br', 'hr', 'link', 'meta', 'source', 'wbr'}
    
    for i, line in enumerate(lines):
        row = i + 1
        # Extract tags from line
        matches = re.finditer(r'<(/?[a-zA-Z0-9\.-]+)([^>]*?)>', line)
        for m in matches:
            tag = m.group(1)
            attrs = m.group(2)
            
            # Simple heuristic to skip Lucide icons and common code snippets that look like tags
            if tag in ['string', 'number', 'T', 'K', 'V', 'File', 'Cut', 'HTMLDivElement', 'Video', 'Film', 'Languages', 'Key', 'Download', 'X', 'Play', 'Pause', 'Scissors', 'Wand2', 'Trash2', 'Loader2']:
                continue
                
            if tag.startswith('/'):
                tag_name = tag[1:]
                if not stack:
                    print(f"[{row}] Error: Extra close tag </{tag_name}>")
                else:
                    last = stack.pop()
                    if last != tag_name:
                        print(f"[{row}] Mismatch: open={last}, close={tag_name}")
            else:
                # Is it self-closing?
                if attrs.strip().endswith('/') or tag.lower() in void_elements:
                    continue
                stack.append(tag)
        
        # print(f"[{row}] Depth: {len(stack)} {stack[-3:] if stack else ''}")

    if stack:
        print(f"Final unclosed stack: {stack}")
    else:
        print("Success: Tags are balanced!")

check_structure('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx')
