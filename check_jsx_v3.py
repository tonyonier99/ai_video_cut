
import re

def check_jsx(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'\{/\*.*?\*/\}', '', content)
    content = re.sub(r'//.*', '', content)

    # Find all tags, but skip self-closing ones like <Tag />
    # This regex is a bit more complex.
    # We want to find <TagName ... but NOT if it ends in />
    
    void_elements = {'img', 'input', 'br', 'hr', 'link', 'meta', 'source', 'wbr'}
    
    # Let's find all matches of <something... > or </something>
    matches = re.finditer(r'<(/?[a-zA-Z0-9\.-]+)([^>]*?)>', content)
    
    stack = []
    for m in matches:
        tag_content = m.group(1)
        attributes = m.group(2)
        
        if tag_content.startswith('/'):
            close_tag = tag_content[1:]
            if not stack:
                print(f"Error: Founding closing tag </{close_tag}> but stack is empty at {m.start()}")
                continue
            last = stack.pop()
            if last != close_tag:
                print(f"Error: Mismatched tag at {m.start()}. Expected </{last}> but found </{close_tag}>")
        else:
            open_tag = tag_content
            # Check if it's self-closing
            if attributes.strip().endswith('/') or open_tag.lower() in void_elements:
                continue
            
            # Simple heuristic for types in code (e.g. <HTMLDivElement>)
            if open_tag in ['string', 'number', 'boolean', 'any', 'void', 'T', 'K', 'V', 'HTMLDivElement', 'HTMLVideoElement', 'WaveSurfer', 'HTMLInputElement', 'File', 'Cut']:
                continue
                
            stack.append(open_tag)
    
    if stack:
        print(f"Error: Unclosed tags: {stack}")
    else:
        print("JSX tags are balanced!")

check_jsx('/Users/huangguanting/.gemini/antigravity/scratch/antigravity-cut/src/App.tsx')
