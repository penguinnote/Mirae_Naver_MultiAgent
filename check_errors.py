import json

try:
    with open('base.json', encoding='utf-8') as f:
        data = json.load(f)
    
    # base.json이 리스트 형태든, 딕셔너리 형태든 모두 처리
    items = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
    
    for v in items:
        if isinstance(v, dict) and v.get('question_id') in ['V-04', 'V-14', 'V-17']:
            print(f"\n==================================================")
            print(f"[{v.get('question_id')}] {v.get('question')}")
            print(f"--------------------------------------------------")
            print(f"▶ 생성된 답변:\n{v.get('answer')}")
            print(f"--------------------------------------------------")
            print(f"▶ 검색된 근거:\n{v.get('retrieved_context')}")
            print(f"==================================================\n")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
