import json
import os
import glob
import re
from tqdm import tqdm

def extract_answer(response, dataset_type):
    """응답에서 답변 추출"""
    if not response:
        return ""
    
    # StrategyQA 처리 (yes/no 답변)
    if 'strategyqa' in dataset_type:
        if re.search(r'the (?:correct )?answer is\s*:?\s*yes', response, re.IGNORECASE):
            return "true"
        elif re.search(r'the (?:correct )?answer is\s*:?\s*no', response, re.IGNORECASE):
            return "false"
        return ""
    
    # 일반적인 경우 (A, B, C, D, E 또는 숫자 답변)
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is \\\( \\\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?) \\\)',
        r'the (?:correct )?answer is\s*:\s*[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*:[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return ""

def get_correct_answer(item):
    """우선순위에 따라 정답을 추출"""
    # 우선순위 1: correct_answer
    if "correct_answer" in item and item["correct_answer"]:
        return str(item["correct_answer"])
    
    # 우선순위 2: original_answer
    if "original_answer" in item and item["original_answer"]:
        return str(item["original_answer"])
    
    # 우선순위 3: answerKey
    if "answerKey" in item and item["answerKey"]:
        return str(item["answerKey"])
    
    # 우선순위 4: answer (StrategyQA에서 사용)
    if "answer" in item and item["answer"] is not None:
        return str(item["answer"]).lower()
    
    return ""

def get_question_text(item, dataset_type):
    """데이터셋 유형에 따라 question 텍스트 추출"""
    # SVAMP 특수 처리: body + question
    if 'svamp' in dataset_type.lower():
        body = item.get('body', '')
        question = item.get('question', '')
        
        # body와 question이 모두 있는 경우 결합
        if body and question:
            # 간단한 구분자로 결합 (필요에 따라 조정)
            return f"{body} {question}"
    
    # 일반적인 경우
    return item.get('question', '')

def create_qa_dataset(input_dir, output_file):
    """QA 데이터셋 생성 (조건부 저장, SVAMP 특수 처리)"""
    dataset = []
    correct_count = 0
    incorrect_count = 0
    total_processed = 0
    
    # 모든 JSON 파일 찾기 (벤치마크 구분 없이)
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    print(f"처리할 파일 수: {len(json_files)}")
    
    for file_path in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            # 파일 이름에서 데이터셋 유형 추출
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            # 데이터셋 유형 (폴더 이름에서 _train 제거)
            dataset_type = dir_name.replace('_train', '')
            
            # 모델명 추출
            model_name = file_name.split('_')[1] if '_' in file_name else "unknown"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 각 항목 처리
            for item in data:
                # 데이터셋 유형에 맞게 question 텍스트 추출
                question = get_question_text(item, dataset_type)
                
                # 정답 정보 추출 (우선순위 적용)
                correct_answer = get_correct_answer(item)
                
                # 응답 처리 (모든 가능한 응답 형식 확인)
                responses_to_process = []
                
                # 개별 응답 항목 (response_1, response_2 등)
                for key in item:
                    if key.startswith('response_') and item[key] and item[key] != 'ERROR':
                        responses_to_process.append(item[key])
                
                # all_responses 배열이 있는 경우
                if 'all_responses' in item and isinstance(item['all_responses'], list):
                    for resp in item['all_responses']:
                        if resp and resp != 'ERROR':
                            responses_to_process.append(resp)
                
                # 단일 response 항목이 있는 경우
                if 'response' in item and item['response'] and item['response'] != 'ERROR':
                    responses_to_process.append(item['response'])
                
                # 각 응답에 대해 처리
                for response in responses_to_process:
                    total_processed += 1
                    
                    # 응답에서 답변 추출 (is_correct 계산용)
                    extracted_answer = extract_answer(response, dataset_type)
                    
                    # 정답 여부 확인
                    is_correct = False
                    if extracted_answer and correct_answer:
                        # StrategyQA의 경우 true/false 비교
                        if 'strategyqa' in dataset_type:
                            is_correct = extracted_answer.lower() == correct_answer.lower()
                        else:
                            # 일반적인 경우 대소문자 무시하고 비교
                            is_correct = extracted_answer.upper() == correct_answer.upper()
                    
                    # 데이터셋 항목 생성 (조건부)
                    dataset_item = {
                        'model': model_name,
                        'dataset_type': dataset_type,
                        'response': response,
                        'is_correct': is_correct
                    }
                    
                    # 정답인 경우에만 question 추가
                    if is_correct:
                        correct_count += 1
                        dataset_item['question'] = question
                    else:
                        incorrect_count += 1
                    
                    dataset.append(dataset_item)
                    
        except Exception as e:
            print(f"파일 처리 오류 ({file_path}): {str(e)}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"데이터셋 생성 완료: {len(dataset)}개 항목 저장")
    print(f"정답 항목: {correct_count}개 (question + response)")
    print(f"오답 항목: {incorrect_count}개 (response만)")
    print(f"정답 비율: {correct_count/total_processed*100:.2f}%")
    
    return dataset

if __name__ == "__main__":
    # 입력 디렉토리와 출력 파일 경로
    input_dir = "/data3/jykim/Projects/CCQA_official/finetuning/train_set"
    output_file = "/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset.json"
    
    # 데이터셋 생성
    create_qa_dataset(input_dir, output_file)