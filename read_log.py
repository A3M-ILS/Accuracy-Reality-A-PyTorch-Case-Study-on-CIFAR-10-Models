import sys
try:
    with open('build2.log', 'rb') as f:
        content = f.read()
        
    try:
        text = content.decode('utf-16le', errors='replace')
    except:
        text = content.decode('utf-8', errors='replace')

    lines = text.split('\n')
    
    with open('clean2.log', 'w', encoding='utf-8') as fw:
        for i, line in enumerate(lines):
            if 'error' in line.lower() or 'failed' in line.lower() or 'not found' in line.lower() or 'canceled' in line.lower():
                start = max(0, i-10)
                end = min(len(lines), i+10)
                fw.write("=== ERROR REGION ===\n")
                fw.write('\n'.join(lines[start:end]))
                fw.write("\n=== END REGION ===\n\n")
    print("Found potential errors. Check clean2.log.")
except Exception as e:
    print(f"Error: {e}")
