import os
import json
import re
import glob
from collections import Counter
from tqdm import tqdm
import csv

def extract_numerical_answer(response):
    """
    답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자를 추출하는 함수
    """
    if not response:
        return None
    
    # Common answer patterns
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    
    # 텍스트를 소문자로 변환
    text = response.lower()
    
    # 패턴들을 순차적으로 검사
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # 캡처된 그룹(숫자) 추출
            answer = match.group(1).strip()
            # 숫자에 천 단위 구분 콤마(,)가 있으면 제거
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    
    return None

def is_answer_correct(extracted_answer, correct_answer):
    """
    추출된 답변과 정답이 일치하는지 확인하는 함수
    """
    if extracted_answer is None or correct_answer is None:
        return False
    
    # 숫자만 추출하여 비교
    extracted_numeric = re.sub(r'[^\d.]', '', str(extracted_answer).strip())
    correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
    
    try:
        if extracted_numeric and correct_numeric:
            extracted_float = float(extracted_numeric)
            correct_float = float(correct_numeric)
            return abs(extracted_float - correct_float) < 1e-5
        else:
            # 숫자가 아닌 경우 문자열 비교
            return str(extracted_answer).strip() == str(correct_answer).strip()
    except ValueError:
        # 숫자로 변환할 수 없는 경우 단순 문자열 비교
        return str(extracted_answer).strip() == str(correct_answer).strip()

def apply_original_self_consistency(results):
    """
    기존 self-consistency 로직을 구현한 함수 - 동률 시 첫 번째 응답을 선택
    """
    updated_results = []
    
    for item in tqdm(results, desc="기존 self-consistency 적용 중", leave=False):
        updated_item = item.copy()
        
        # 모든 response_n 키 찾기
        responses = []
        for i in range(1,6):  # 최대 5개까지 검사
            response_key = f"response_{i}"
            if response_key in item and item[response_key]:
                responses.append(item[response_key])
            else:
                break
                
        if not responses:
            updated_item["original_sc_answer"] = None
            updated_results.append(updated_item)
            continue
            
        # 각 응답에서 정답 추출
        extracted_answers = []
        for response in responses:
            answer = extract_numerical_answer(response)
            extracted_answers.append(answer)
        
        # 정답 빈도 계산
        answer_counter = Counter([ans for ans in extracted_answers if ans is not None])
        
        if not answer_counter:
            # 추출된 정답이 없는 경우
            updated_item["original_sc_answer"] = None
        else:
            # 가장 많은 표를 받은 정답 찾기
            most_common_answers = answer_counter.most_common()
            top_answer = most_common_answers[0][0]
            updated_item["original_sc_answer"] = top_answer
        
        updated_results.append(updated_item)
    
    return updated_results

def apply_new_self_consistency(results):
    """
    새로운 self-consistency 적용:
    1. 모든 응답에 동일한 가중치(1)를 부여
    2. 최다 득표 답변의 출현 횟수가 2개 이하인 경우 NLI 점수로 결정
    """
    updated_results = []
    
    for item in tqdm(results, desc="새로운 self-consistency 적용 중"):
        updated_item = item.copy()
        
        # 각 응답에서 정답 추출
        extracted_answers = []
        
        for i in range(1, 6):
            response_key = f"response_{i}"
            if response_key in item and item[response_key]:
                answer = extract_numerical_answer(item[response_key])
                extracted_answers.append((i, answer))  # 인덱스와 함께 저장
            else:
                break
        
        # 유효한 답변만 필터링 (None이 아닌 것들)
        valid_answers = [(idx, ans) for idx, ans in extracted_answers if ans is not None]
        
        if not valid_answers:
            # 추출된 정답이 없는 경우
            updated_item["ccqa_answer"] = None
            updated_item["ccqa_method"] = "none"
            updated_results.append(updated_item)
            continue
            
        # 답변 카운트
        answer_counter = Counter([ans for _, ans in valid_answers])
        most_common_answers = answer_counter.most_common()
        
        if not most_common_answers:
            # 추출된 정답이 없는 경우
            updated_item["ccqa_answer"] = None
            updated_item["ccqa_method"] = "none"
        else:
            # 가장 많은 표를 얻은 답변 확인
            best_answer = most_common_answers[0][0]
            best_count = most_common_answers[0][1]
            
            # 최다 득표 답변의 개수가 2개 이하면 precision 점수로 결정
            if best_count <= 1:
                # NLI 점수로 결정하기
                all_nli_scores = []
                
                for idx, ans in valid_answers:
                    if "similarity_scores" in item and f"question_{idx}" in item["similarity_scores"]:
                        # precision 점수 가져오기
                        nli_score = item["similarity_scores"].get(f"question_{idx}", 0)
                        all_nli_scores.append((ans, idx, nli_score))
                
                if all_nli_scores:
                    # precision 점수 순으로 정렬
                    all_nli_scores.sort(key=lambda x: x[2], reverse=True)
                    best_answer, best_idx, best_score = all_nli_scores[0]
                    
                    updated_item["ccqa_answer"] = best_answer
                    updated_item["ccqa_method"] = "highest_precision_score"
                else:
                    # NLI 점수가 없는 경우 다수결 결과 사용
                    updated_item["ccqa_answer"] = best_answer
                    updated_item["ccqa_method"] = "majority_vote_no_nli"
            else:
                # 득표 개수가 3개 이상이면 다수결 결과 사용
                updated_item["ccqa_answer"] = best_answer
                updated_item["ccqa_method"] = "majority_vote"
        
        updated_results.append(updated_item)
    
    return updated_results

def extract_model_name(filename):
    """
    파일명에서 모델 이름을 추출하는 함수
    """
    patterns = [r'GSM8K_([^_]+)_']
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # 패턴이 맞지 않으면 파일명에서 확장자만 제거하여 반환
    return os.path.splitext(filename)[0]

def process_file(file_path):
    """
    단일 파일 처리 및 결과 생성
    """
    try:
        # 파일명에서 모델명 추출
        filename = os.path.basename(file_path)
        model_name = extract_model_name(filename)
        
        # 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 결과 항목 가져오기
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            time_info = data.get("time_info", {})
        else:
            results = data
            time_info = {}
        
        # 기존 self-consistency 적용
        results_with_original_sc = apply_original_self_consistency(results)
        
        # 새로운 self-consistency 적용
        processed_results = apply_new_self_consistency(results_with_original_sc)
        
        # 정확도 계산
        total_items = len(processed_results)
        original_sc_correct_count = 0
        ccqa_correct_count = 0
        
        for item in processed_results:
            correct_answer = item.get("original_answer")
            
            # 1. 기존 SC 정확도 계산
            original_sc_answer = item.get("original_sc_answer")
            if original_sc_answer is not None and correct_answer is not None:
                if is_answer_correct(original_sc_answer, correct_answer):
                    original_sc_correct_count += 1
                    item["original_sc_is_correct"] = True
                else:
                    item["original_sc_is_correct"] = False
            else:
                item["original_sc_is_correct"] = False
            
            # 2. 새로운 CCQA 정확도 계산
            ccqa_answer = item.get("ccqa_answer")
            if ccqa_answer is not None and correct_answer is not None:
                if is_answer_correct(ccqa_answer, correct_answer):
                    ccqa_correct_count += 1
                    item["ccqa_is_correct"] = True
                else:
                    item["ccqa_is_correct"] = False
            else:
                item["ccqa_is_correct"] = False
        
        # 정확도 계산
        original_sc_accuracy = original_sc_correct_count / total_items if total_items > 0 else 0
        ccqa_accuracy = ccqa_correct_count / total_items if total_items > 0 else 0
        
        # 처리 결과를 파일로 저장
        output_dir_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), "ccqa_results")
        os.makedirs(output_dir_path, exist_ok=True)
        output_filename = os.path.basename(file_path).replace('.json', '_ccqa.json')
        output_file_path = os.path.join(output_dir_path, output_filename)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": processed_results,
                "model_name": model_name,
                "time_info": time_info
            }, f, indent=2)
        
        print(f"{model_name}: 총 {total_items}개 문제")
        print(f"  - 기존 SC 정확도: {original_sc_accuracy:.2%} ({original_sc_correct_count}개 정답)")
        print(f"  - CCQA 정확도: {ccqa_accuracy:.2%} ({ccqa_correct_count}개 정답)")
        
        # 방법별 카운트
        method_counts = Counter([item.get("ccqa_method", "unknown") for item in processed_results])
        for method, count in method_counts.items():
            print(f"  - {method}: {count}개 문제 ({count/total_items:.2%})")
        
        # 방법별 정확도
        method_accuracy = {}
        for method in method_counts.keys():
            method_items = [item for item in processed_results if item.get("ccqa_method") == method]
            correct_items = [item for item in method_items if item.get("ccqa_is_correct", False)]
            method_accuracy[method] = len(correct_items) / len(method_items) if method_items else 0
        
        return {
            "model_name": model_name,
            "total_items": total_items,
            "original_sc_accuracy": original_sc_accuracy,
            "original_sc_correct_count": original_sc_correct_count,
            "ccqa_accuracy": ccqa_accuracy,
            "ccqa_correct_count": ccqa_correct_count,
            "method_counts": dict(method_counts),
            "method_accuracy": method_accuracy
        }
    
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison_csv(all_results, output_dir):
    """
    모든 모델에 대한 비교 결과를 CSV로 생성 - 간소화 버전
    """
    # 결과 요약 CSV 생성
    csv_path = os.path.join(output_dir, "ccqa_comparison.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # CSV 헤더 정의 - 간소화
        fieldnames = [
            'model_name', 
            'original_sc_accuracy',
            'ccqa_accuracy',
            'improvement',
            'total_items'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 각 결과 정보 작성
        for result in all_results:
            # 성능 향상 계산
            improvement = result['ccqa_accuracy'] - result['original_sc_accuracy']
            
            row = {
                'model_name': result['model_name'],
                'original_sc_accuracy': f"{result['original_sc_accuracy']:.4f}",
                'ccqa_accuracy': f"{result['ccqa_accuracy']:.4f}",
                'improvement': f"{improvement:.4f}",
                'total_items': result['total_items']
            }
            
            writer.writerow(row)
    
    print(f"\n모든 처리가 완료되었습니다.")
    print(f"간소화된 비교 결과는 {csv_path}에 저장되었습니다.")

def main():
    """
    메인 함수 - 여러 폴더의 모든 JSON 파일 처리
    """
    # 입력 폴더 경로
    INPUT_FOLDER = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar"
    
    # 출력 폴더 경로
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar/tie_voting_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    
    if not json_files:
        print(f"경고: {INPUT_FOLDER}에서 JSON 파일을 찾을 수 없습니다!")
        return
    
    print(f"총 {len(json_files)}개 파일을 처리합니다.")
    
    # 모든 파일 처리 결과 저장
    all_results = []
    
    # 각 파일 처리
    for file_path in tqdm(json_files, desc="파일 처리 중"):
        result = process_file(file_path)
        if result:
            all_results.append(result)
    
    # 결과 비교 CSV 생성 - 간소화 버전
    create_comparison_csv(all_results, OUTPUT_DIR)

if __name__ == "__main__":
    main()