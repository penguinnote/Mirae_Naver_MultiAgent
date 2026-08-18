import json

def search_target(data, targets):
    if isinstance(data, dict):
        if data.get('question_id') in targets:
            print(f"\n==================================================")
            print(f"[{data.get('question_id')}] {data.get('question')}")
            print(f"--------------------------------------------------")
            print(f"▶ 생성된 답변:\n{data.get('answer')}")
            print(f"--------------------------------------------------")
            print(f"▶ 검색된 근거:\n{data.get('retrieved_context')}")
            print(f"==================================================\n")
        for v in data.values():
            search_target(v, targets)
    elif isinstance(data, list):
        for item in data:
            search_target(item, targets)

try:
    with open('base.json', 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content.startswith('[') and not content.startswith('{'):
            for line in content.split('\n'):
                if line.strip():
                    search_target(json.loads(line), ['V-04', 'V-14', 'V-17'])
        else:
            search_target(json.loads(content), ['V-04', 'V-14', 'V-17'])
except Exception as e:
    print(f"오류: {e}")
