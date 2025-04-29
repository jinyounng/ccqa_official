import os
import json
import re
import random
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
import glob

def extract_numerical_answer(response: str) -> Optional[str]:
    """
    답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자를 추출하는 함수
    """
    if not response:
        return None
    
    # 수학 문제 정답 패턴
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    
    # 전체 텍스트를 소문자로 변환
    text = response.lower()
    
    # 패턴들을 순차적으로 검사하면서 가장 먼저 발견된 결과 반환
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

def apply_original_self_consistency(results: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    기존 self-consistency 로직을 구현한 함수 - 동률 시 랜덤으로 하나를 선택하도록 수정
    
    Args:
        results: 원본 결과 데이터 리스트
        seed: 랜덤 시드
        
    Returns:
        self-consistency가 적용된 결과 데이터 리스트
    """
    # 랜덤 시드 설정
    random.seed(seed)
    
    updated_results = []
    
    for item in results:
        updated_item = item.copy()
        
        # 모든 response_n 키 찾기
        responses = []
        for i in range(1, 6):  # 최대 5개까지 검사
            response_key = f"response_{i}"
            if response_key in item and item[response_key]:
                responses.append(item[response_key])
            else:
                break
                
        if not responses:
            updated_item["self_consistency_answer"] = None
            updated_item["self_consistency_extraction"] = []
            updated_item["self_consistency_vote"] = {}
            updated_results.append(updated_item)
            continue
            
        # 각 응답에서 정답 추출
        extracted_answers = []
        for response in responses:
            answer = extract_numerical_answer(response)
            extracted_answers.append(answer)
        
        # 결과 저장
        updated_item["self_consistency_extraction"] = extracted_answers
        
        # 정답 빈도 계산
        answer_counter = Counter([ans for ans in extracted_answers if ans is not None])
        
        if not answer_counter:
            # 추출된 정답이 없는 경우
            updated_item["self_consistency_answer"] = None
            updated_item["self_consistency_vote"] = {}
        else:
            # 가장 많은 표를 받은 정답 찾기
            most_common_answers = answer_counter.most_common()
            max_count = most_common_answers[0][1]
            
            # 동일한 득표수를 가진 정답들 찾기
            top_answers = [ans for ans, count in most_common_answers if count == max_count]
            
            if len(top_answers) == 1:
                # 최다 득표가 하나인 경우
                updated_item["self_consistency_answer"] = top_answers[0]
            else:
                # 동률인 경우 랜덤으로 선택
                updated_item["self_consistency_answer"] = random.choice(top_answers)
                updated_item["self_consistency_tie_breaker"] = "random"  # 동률 처리 방법 기록
            
            # 투표 결과 저장
            updated_item["self_consistency_vote"] = {ans: count for ans, count in most_common_answers}
        
        updated_results.append(updated_item)
    
    return updated_results

def is_numeric_answer_correct(predicted: str, correct: str) -> bool:
    """
    수치형 예측 답변이 정답과 일치하는지 확인하는 함수
    """
    if predicted is None or correct is None:
        return False
    
    # 숫자만 추출하여 비교
    predicted_numeric = re.sub(r'[^\d.]', '', str(predicted).strip())
    correct_numeric = re.sub(r'[^\d.]', '', str(correct).strip())
    
    try:
        if predicted_numeric and correct_numeric:
            predicted_float = float(predicted_numeric)
            correct_float = float(correct_numeric)
            return abs(predicted_float - correct_float) < 1e-5
        else:
            # 숫자가 아닌 경우 문자열 비교
            return str(predicted).strip() == str(correct).strip()
    except ValueError:
        # 숫자로 변환할 수 없는 경우 단순 문자열 비교
        return str(predicted).strip() == str(correct).strip()

def calculate_sc_accuracy_for_seed(file_path: str, seed: int) -> float:
    """
    특정 시드에 대한 SC 정확도를 계산하는 함수
    
    Args:
        file_path: 입력 파일 경로
        seed: 랜덤 시드
    
    Returns:
        SC 정확도
    """
    try:
        # 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 결과 항목 가져오기
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = data
        
        # 처리용 결과 복사
        processed_results = []
        
        # 각 항목 기본 처리
        for item in results:
            processed_item = item.copy()
            
            # 정답 필드 확인
            correct_answer = None
            if "correct_answer" in processed_item:
                correct_answer = processed_item["correct_answer"]
            elif "original_answer" in processed_item:
                correct_answer = processed_item["original_answer"]
            
            if correct_answer:
                processed_item["correct_answer"] = correct_answer
            
            processed_results.append(processed_item)
        
        # Self-consistency 적용
        sc_results = apply_original_self_consistency(processed_results, seed=seed)
        
        # Self-consistency 정확도 계산
        sc_correct_count = 0
        total_items = len(sc_results)
        
        for item in sc_results:
            sc_answer = item.get("self_consistency_answer")
            correct_answer = item.get("correct_answer")
            
            if sc_answer is not None and correct_answer is not None:
                if is_numeric_answer_correct(sc_answer, correct_answer):
                    sc_correct_count += 1
        
        # Self-consistency 정확도 계산
        sc_accuracy = sc_correct_count / total_items if total_items > 0 else 0
        
        return sc_accuracy
    
    except Exception as e:
        print(f"Error processing file {file_path} with seed {seed}: {str(e)}")
        return 0.0

def analyze_file_seeds(file_path: str, start_seed: int = 1, end_seed: int = 100) -> Dict:
    """
    하나의 파일에 대해 모든 시드의 SC 정확도를 계산하는 함수
    
    Args:
        file_path: 입력 파일 경로
        start_seed: 시작 시드
        end_seed: 종료 시드
    
    Returns:
        시드별 정확도 결과
    """
    file_name = os.path.basename(file_path)
    print(f"\n파일 분석 중: {file_name}")
    
    # 결과 저장 구조
    seed_results = {}
    
    # 모든 시드에 대해 반복
    for seed in tqdm(range(start_seed, end_seed + 1), desc=f"Analyzing seeds for {file_name}"):
        accuracy = calculate_sc_accuracy_for_seed(file_path, seed)
        seed_results[seed] = accuracy
    
    return {
        "file_name": file_name,
        "seed_results": seed_results
    }

def find_worst_seed_range(
    input_folders: List[str], 
    start_seed: int = 1, 
    end_seed: int = 91,  # 범위가 10이므로 최대 시작 시드는 91
    range_size: int = 10,
    min_percentage: float = 50.0
) -> Dict:
    """
    평균 SC 정확도가 가장 낮은 시드 범위를 찾는 함수
    
    Args:
        input_folders: 입력 폴더 목록
        start_seed: 시작 시드
        end_seed: 마지막 시작 시드
        range_size: 각 범위의 크기
        min_percentage: 최소 파일 비율 (%)
    
    Returns:
        평균 정확도가 가장 낮은 시드 범위 정보
    """
    all_file_paths = []
    
    # 모든 파일 경로 수집
    for input_folder in input_folders:
        json_files = glob.glob(os.path.join(input_folder, "*.json"))
        all_file_paths.extend(json_files)
    
    if not all_file_paths:
        print("처리할 JSON 파일을 찾을 수 없습니다!")
        return {}
    
    print(f"총 {len(all_file_paths)}개 파일을 분석합니다.")
    
    # 각 파일별 시드 분석
    all_file_results = []
    
    for file_path in all_file_paths:
        file_result = analyze_file_seeds(file_path, start_seed, end_seed + range_size - 1)
        all_file_results.append(file_result)
    
    # 각 시드 범위의 평균 정확도 계산
    range_accuracies = {}
    
    # 각 가능한 시작 시드에 대해
    for range_start in range(start_seed, end_seed + 1):
        range_end = range_start + range_size - 1
        
        # 각 파일별 이 범위의 평균 정확도 계산
        file_range_accuracies = []
        
        for file_result in all_file_results:
            seed_results = file_result["seed_results"]
            
            # 이 범위의 시드들의 정확도 추출
            range_seeds = [s for s in range(range_start, range_end + 1) if s in seed_results]
            range_seed_accuracies = [seed_results[s] for s in range_seeds]
            
            # 평균 정확도 계산
            if range_seed_accuracies:
                avg_accuracy = sum(range_seed_accuracies) / len(range_seed_accuracies)
                file_range_accuracies.append(avg_accuracy)
        
        # 전체 파일에 대한 이 범위의 평균 정확도
        if file_range_accuracies:
            overall_avg = sum(file_range_accuracies) / len(file_range_accuracies)
            range_accuracies[(range_start, range_end)] = {
                "overall_avg": overall_avg,
                "file_accuracies": file_range_accuracies
            }
    
    # 평균 정확도가 가장 낮은 시드 범위 찾기
    if range_accuracies:
        worst_range = min(range_accuracies.items(), key=lambda x: x[1]["overall_avg"])
        worst_start, worst_end = worst_range[0]
        worst_avg = worst_range[1]["overall_avg"]
        
        print(f"\n평균 SC 정확도가 가장 낮은 시드 범위: {worst_start}~{worst_end}, 평균 정확도: {worst_avg:.4f}")
        
        # 최악의 범위에서 각 시드별 정확도 비교를 위한 데이터 수집
        worst_range_seed_details = []
        
        for seed in range(worst_start, worst_end + 1):
            seed_accuracies = []
            
            # 각 파일에서의 이 시드의 정확도 수집
            for file_result in all_file_results:
                seed_results = file_result["seed_results"]
                if seed in seed_results:
                    seed_accuracies.append(seed_results[seed])
            
            # 이 시드의 평균 정확도
            if seed_accuracies:
                seed_avg = sum(seed_accuracies) / len(seed_accuracies)
                worst_range_seed_details.append({
                    "seed": seed,
                    "avg_accuracy": seed_avg,
                    "accuracies": seed_accuracies
                })
        
        # 시드별 평균 정확도 기준 정렬
        worst_range_seed_details.sort(key=lambda x: x["avg_accuracy"])
        
        print("\n최악 범위 내 각 시드별 평균 정확도 (오름차순):")
        for detail in worst_range_seed_details:
            print(f"시드 {detail['seed']}: 평균 정확도 {detail['avg_accuracy']:.4f}")
        
        if worst_range_seed_details:
            absolute_worst_seed = worst_range_seed_details[0]["seed"]
            absolute_worst_avg = worst_range_seed_details[0]["avg_accuracy"]
            print(f"\n범위 내에서 가장 성능이 낮은 단일 시드: {absolute_worst_seed}, 평균 정확도: {absolute_worst_avg:.4f}")
        
        result = {
            "worst_range_start": worst_start,
            "worst_range_end": worst_end,
            "worst_range_avg": worst_avg,
            "worst_range_seed_details": worst_range_seed_details,
            "all_range_accuracies": range_accuracies
        }
        
        return result
    else:
        print("평균 정확도 계산에 실패했습니다.")
        return {}

def main():
    """
    메인 함수
    """
    # 입력 폴더 목록
    INPUT_FOLDERS = [
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar",
    ]
    
    # 시드 범위 및 분석 설정
    START_SEED = 1
    END_SEED = 91  # 범위가 10이므로 최대 시작 시드는 91
    RANGE_SIZE = 10  # 각 범위의 크기
    MIN_PERCENTAGE = 70.0  # 최소 파일 비율 (%)
    
    # 평균 SC 정확도가 가장 낮은 시드 범위 찾기
    results = find_worst_seed_range(
        input_folders=INPUT_FOLDERS,
        start_seed=START_SEED,
        end_seed=END_SEED,
        range_size=RANGE_SIZE,
        min_percentage=MIN_PERCENTAGE
    )
    
    # 결과 저장 (선택 사항)
    # output_file = "worst_seed_range_results.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2)
    # print(f"\n결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()