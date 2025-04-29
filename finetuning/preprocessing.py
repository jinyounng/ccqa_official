import os
import json
import re

folder_path = '/data3/jykim/Projects/CCQA_official/finetuning/train_set/svamp_train'
# 답변 추출을 위한 정규식 패턴들
patterns = [
    r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)', 
    r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
    r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
    r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    r'the (?:correct )?answer is \\\( \\\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?) \\\)',
    r'the (?:correct )?answer is\s*:\s*[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
    r'the (?:correct )?answer is\s*:[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
]

# 추가적인 답변 형식 패턴
additional_patterns = [
    r'so the answer is[^.]*\.?',
    r'therefore,? the answer is[^.]*\.?',
    r'thus,? the answer is[^.]*\.?',
    r'hence,? the answer is[^.]*\.?',
    r'in conclusion,? the answer is[^.]*\.?',
    r'finally,? the answer is[^.]*\.?',
    r'we (?:can |now )?conclude that[^.]*\.?',
    r'the answer to this question is[^.]*\.?',
    r'my final answer is[^.]*\.?',
    r'the solution is[^.]*\.?',
    r'the result is[^.]*\.?',
    r'i conclude that[^.]*\.?',
]

# 질문 패턴 (다양한 형식 포함)
question_patterns = [
    r'\nQ[.:]\s',     # \nQ. 또는 \nQ:에 공백이 있는 경우
    r'\nQ[.:]',       # \nQ. 또는 \nQ:에 공백이 없는 경우
    r'\n\s*Q[.:]\s',  # 줄바꿈 후 공백이 있고 Q. 또는 Q:가 오는 경우
    r'\nQuestion:',   # \nQuestion: 패턴
    r'\nProblem:',    # \nProblem: 패턴
    r'<\|eot_id\|>',  # <|eot_id|> 패턴 (종료 태그)
]

def process_text_content(text):
    """텍스트 내용 처리: 'the answer is' 등의 패턴이 나오면 그 이후의 모든 내용 삭제"""
    answer_found = False
    answer = None
    match_start_index = -1
    
    # 주요 패턴에 대해 검사 (캡처 그룹 포함 패턴)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1)  # 첫 번째 캡처 그룹이 실제 답변
            match_start_index = match.start()
            answer_found = True
            break
    
    # 패턴을 찾지 못한 경우 추가 패턴 검사 (캡처 그룹 없는 패턴)
    if not answer_found:
        for pattern in additional_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                match_start_index = match.start()
                answer_found = True
                break
    
    if answer_found:
        # 패턴이 발견된 위치까지만 텍스트 유지
        text = text[:match_start_index].strip()
        
        # 텍스트가 너무 짧게 잘리는 것을 방지하기 위한 최소 길이 체크
        if len(text) < 50:  # 최소 50자 이상 유지
            return text  # 원본 텍스트 반환
    
    # 질문 패턴 검색 및 처리
    cut_indices = []
    for pattern in question_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # 첫 번째 질문이 아닌 다음 질문부터 자름
            if len(matches) > 1:
                cut_indices.append(matches[1].start())
    
    # 다음 질문이 있으면 해당 위치에서 텍스트 자르기
    if cut_indices:
        first_question_index = min(cut_indices)
        text = text[:first_question_index]
    
    return text.strip()

def process_json_content(json_content):
    """JSON 배열 내용 처리"""
    try:
        # JSON 배열 또는 단일 객체 파싱
        data = json.loads(json_content)
        
        # 배열인 경우
        if isinstance(data, list):
            for item in data:
                if 'response_1' in item:
                    item['response_1'] = process_text_content(item['response_1'])
        # 단일 객체인 경우
        elif isinstance(data, dict) and 'response_1' in data:
            data['response_1'] = process_text_content(data['response_1'])
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print(f'JSON 파싱 오류: {e}')
        return json_content  # 오류 발생 시 원본 반환

def process_all_json_files():
    """폴더 내 모든 JSON 파일 처리"""
    try:
        # 폴더 내 모든 파일 목록 가져오기
        files = os.listdir(folder_path)
        
        # JSON 파일만 필터링
        json_files = [file for file in files if file.lower().endswith('.json')]
        
        print(f'처리할 JSON 파일 수: {len(json_files)}')
        
        # 각 JSON 파일 처리
        processed_count = 0
        for file in json_files:
            file_path = os.path.join(folder_path, file)
            
            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                
                # JSON 내용 처리
                processed_data = process_json_content(data)
                
                # 처리된 내용 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_data)
                
                processed_count += 1
                print(f'{file} 처리 완료 ({processed_count}/{len(json_files)})')
            
            except Exception as e:
                print(f'{file} 처리 중 오류 발생: {e}')
        
        print('모든 파일 처리 완료')
    
    except Exception as e:
        print(f'처리 중 오류 발생: {e}')

# 예시 테스트
def test_with_example():
    example = '''[
  {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\\n#### 72",
    "extracted_answer": 72.0,
    "correct_answer": "72",
    "response_1": "In April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. In April, she sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. In April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nThe answer is 72. In April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nThe answer is 72. \\nThe answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. The total number of clips she sold is 48 + 24 = 72. The answer is 72. \\nThe answer is 72. \\nThe answer is 72. \\nThe answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nIn April, Natalia sold 48 clips. In May she sold 24. The answer is 72. \\nThe answer is 72. \\nThe answer is"
  }
]'''
    
    example2 = '''[
  {
    "question": "Is the sum of 5 and 7 greater than the product of 2 and 5?",
    "response_1": "To solve this problem, I need to find the sum of 5 and 7, and compare it with the product of 2 and 5. Let me calculate both values:\n\nSum of 5 and 7 = 5 + 7 = 12\nProduct of 2 and 5 = 2 × 5 = 10\n\nNow I can compare: Is 12 > 10?\nYes, 12 is greater than 10.\n\nTherefore, the answer is yes, the sum of 5 and 7 is greater than the product of 2 and 5."
  }
]'''
    
    example3 = '''[
  {
    "question": "What is the capital of France?",
    "response_1": "The capital of France is Paris. This has been the capital city for centuries and is known for landmarks like the Eiffel Tower and the Louvre Museum. So the answer is Paris."
  }
]'''
    
    print("원본 JSON 예시 1:")
    print(example[:500] + "...")
    processed = process_json_content(example)
    print("\n처리된 JSON 예시 1:")
    print(processed[:500] + "...")
    
    # 처리된 내용에서 response_1 추출
    processed_data = json.loads(processed)
    if processed_data and isinstance(processed_data, list) and processed_data[0].get('response_1'):
        print("\nresponse_1 결과 예시 1:")
        print(processed_data[0]['response_1'])
    
    print("\n원본 JSON 예시 2:")
    print(example2)
    processed2 = process_json_content(example2)
    print("\n처리된 JSON 예시 2:")
    print(processed2)
    
    print("\n원본 JSON 예시 3:")
    print(example3)
    processed3 = process_json_content(example3)
    print("\n처리된 JSON 예시 3:")
    print(processed3)
    
    return processed

if __name__ == '__main__':
    # 예시 테스트
    test_with_example()
    
    # 실제 파일 처리 (사용시 주석 해제)
    process_all_json_files()