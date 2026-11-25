import json

transcript_path = r'C:\Users\bui toan\.gemini\antigravity\brain\5438a59b-10dc-43f6-8139-9c17d76ca7aa\.system_generated\logs\transcript.jsonl'
found = []
with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content = data.get('content', '')
        if any(w in content.lower() for w in ['kế hoạch', 'bản kế hoạch', 'plan', 'lộ trình']):
            found.append((i, data.get('type'), content[:150]))

print(f'Total matches: {len(found)}')
for idx, t, c in found[:15]:
    print(f'Line {idx} [{t}]: {repr(c)}')
