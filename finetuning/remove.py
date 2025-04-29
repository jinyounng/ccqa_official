import json
import re

def remove_answer_pattern_from_question(text):
    """텍스트에서 '\n\nAnswer:' 패턴이 나오면 해당 위치에서 텍스트 자르기"""
    # 패턴 변형을 모두 고려 (콜론 뒤에 공백이 있을 수도 있고 없을 수도 있음)
    patterns = [
        r'\n\nAnswer:',
        r'\n\nAnswer: ',
        r'\nAnswer:',
        r'\nAnswer: '
    ]
    
    # 첫 번째 패턴 위치 찾기
    earliest_pos = len(text)  # 초기값은 텍스트 끝
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    # 패턴이 발견됐으면 그 위치까지만 유지
    if earliest_pos < len(text):
        return text[:earliest_pos].strip()
    
    # 패턴이 없으면 원본 텍스트 반환
    return text.strip()

def process_file(file_path):
    """단일 JSON 파일 처리"""
    try:
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # JSON 파싱
        data = json.loads(content)
        
        # 변경된 항목 수 카운팅
        modified_count = 0
        
        # 단일 객체인 경우
        if isinstance(data, dict):
            # question 필드에서만 "\n\nAnswer:" 이후 내용 제거
            if 'question' in data:
                original = data['question']
                data['question'] = remove_answer_pattern_from_question(data['question'])
                if original != data['question']:
                    modified_count += 1
                
        # 배열인 경우
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'question' in item:
                    original = item['question']
                    item['question'] = remove_answer_pattern_from_question(item['question'])
                    if original != item['question']:
                        modified_count += 1
        
        # 처리된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"파일 처리 완료: {file_path}")
        print(f"수정된 항목 수: {modified_count}")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 처리할 파일 경로 설정
    file_path = "/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_qwen14B.json"
    
    # 파일 처리 실행
    process_file(file_path)