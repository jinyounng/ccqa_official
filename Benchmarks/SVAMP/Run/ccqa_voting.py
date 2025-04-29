import os
import json
import re
import random
import sys
import csv
import glob
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm
random.seed(5)

def extract_numerical_answer(response: str) -> Optional[str]:
    """
    답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자를 추출하는 함수
    """
    if not response:
        return None
    
    # 수학 문제 정답 패턴
    patterns = [
        r'the answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the answer is\s*(?:\()?([A-Ea-e])(?:\))?',
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
    
    for item in tqdm(results, desc=f"Applying original self-consistency (seed={seed})", leave=False):
        updated_item = item.copy()
        
        # 모든 response_n 키 찾기
        responses = []
        for i in range(1, 6):  # 최대 20개까지 검사
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
        answer_counter = Counter(extracted_answers)
        
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

def apply_weighted_self_consistency(results: List[Dict[str, Any]], weights: List[int], seed: int = 42) -> List[Dict[str, Any]]:
    """
    가중치 기반 self-consistency 적용 함수 - 여러 가중치 설정을 지원하도록 수정
    
    Args:
        results: 원본 결과 데이터 리스트
        weights: 상위 응답들에 적용할 가중치 리스트
        seed: 랜덤 시드
        
    Returns:
        가중치 기반 self-consistency가 적용된 결과 데이터 리스트
    """
    # 랜덤 시드 설정
    random.seed(seed)
    
    updated_results = []
    
    # 가중치 정보를 문자열로 변환하여 필드에 저장
    weight_info = '_'.join(map(str, weights))
    
    for item in tqdm(results, desc=f"Applying weighted SC [{weight_info}] (seed={seed})", leave=False):
        updated_item = item.copy()
        
        # most_similar_idxs가 있는지 확인
        if "most_similar_idxs" not in item or not item["most_similar_idxs"]:
            # most_similar_idxs가 없으면 기존 most_similar_idx 사용
            if "most_similar_idx" in item:
                updated_item[f"weighted_sc_method_{weight_info}"] = "single_idx"
                most_similar_idx = item["most_similar_idx"]
                response_key = f"response_{most_similar_idx}"
                if response_key in item:
                    # 해당 응답에서 정답 추출
                    answer = extract_numerical_answer(item[response_key])
                    updated_item[f"weighted_sc_answer_{weight_info}"] = answer
                    updated_item[f"weighted_sc_source_{weight_info}"] = "single_response"
            else:
                # 아무것도 없으면 첫 번째 응답 사용
                updated_item[f"weighted_sc_method_{weight_info}"] = "fallback_to_first"
                if "response_1" in item:
                    answer = extract_numerical_answer(item["response_1"])
                    updated_item[f"weighted_sc_answer_{weight_info}"] = answer
                    updated_item[f"weighted_sc_source_{weight_info}"] = "first_response"
            
            updated_results.append(updated_item)
            continue
        
        # most_similar_idxs에서 상위 N개 인덱스 가져오기 (순서대로)
        top_indices = item["most_similar_idxs"][:len(weights)]
        
        # 인덱스가 가중치 개수보다 적으면 부족한 만큼 채움
        if len(top_indices) < len(weights):
            # 빠진 인덱스들 찾기 (1-5 중에서)
            missing_indices = [i for i in range(1, 6) if i not in top_indices]
            # 부족한 만큼 채우기
            top_indices.extend(missing_indices[:len(weights)-len(top_indices)])
        
        # 각 응답에서 정답 추출 및 가중치 적용
        answer_votes = []
        extracted_answers = []
        
        # 상위 N개 응답 각각에 가중치 적용
        # 상위 N개 응답 각각에 가중치 적용
        for idx, response_idx in enumerate(top_indices[:len(weights)]):
            response_key = f"response_{response_idx}"
            if response_key in item and item[response_key]:
                answer = extract_numerical_answer(item[response_key])
                extracted_answers.append(answer)
                # if answer 조건 제거 - None도 가중치 적용
                # 해당 인덱스의 가중치만큼 표 부여
                weight = weights[idx]
                answer_votes.extend([answer] * weight)
        
        # 정답 투표 결과 저장
        updated_item[f"weighted_vote_indices_{weight_info}"] = top_indices
        updated_item[f"weighted_vote_answers_{weight_info}"] = answer_votes
        updated_item[f"weighted_vote_weights_{weight_info}"] = weights  # 사용된 가중치 체계 저장
        updated_item[f"weighted_vote_extracted_{weight_info}"] = extracted_answers  # 추출된 원본 답변 저장
        
        # 투표 결과가 있으면 다수결로 결정
        if answer_votes:
            # 투표 수 계산
            vote_counts = Counter(answer_votes)
            # 가장 많은 표를 받은 정답
            most_common_answers = vote_counts.most_common()
            
            # 최다 득표 정답 선택
            top_answer, top_count = most_common_answers[0]
            
            # 동률이 있는지 확인
            tied_answers = [answer for answer, count in most_common_answers if count == top_count]
            
            if len(tied_answers) > 1:
                # 동률인 경우 랜덤으로 선택
                selected_answer = random.choice(tied_answers)
                updated_item[f"weighted_sc_answer_{weight_info}"] = selected_answer
                updated_item[f"weighted_sc_source_{weight_info}"] = f"random_of_tied_{len(tied_answers)}"
                updated_item[f"weighted_sc_method_{weight_info}"] = "weighted_vote_with_tie"
                updated_item[f"weighted_sc_tie_breaker_{weight_info}"] = "random"  # 동률 처리 방법 기록
            else:
                # 동률이 없으면 최다 득표 정답 선택
                updated_item[f"weighted_sc_answer_{weight_info}"] = top_answer
                updated_item[f"weighted_sc_source_{weight_info}"] = f"votes_{top_count}_of_{sum(vote_counts.values())}"
                updated_item[f"weighted_sc_method_{weight_info}"] = "weighted_vote"
        else:
            # 투표 결과가 없으면 첫 번째 응답 사용
            updated_item[f"weighted_sc_method_{weight_info}"] = "fallback_to_first"
            if "response_1" in item:
                answer = extract_numerical_answer(item["response_1"])
                updated_item[f"weighted_sc_answer_{weight_info}"] = answer
                updated_item[f"weighted_sc_source_{weight_info}"] = "first_response"
        
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

def process_single_file(file_path: str, weight_schemes: List[List[int]], seed: int) -> Dict:
    """
    단일 파일을 처리하여 CoT, Self-Consistency, 여러 가중치 기반 Self-Consistency를 적용합니다.
    
    Args:
        file_path: 입력 파일 경로
        weight_schemes: 여러 가중치 설정들의 리스트
        seed: 랜덤 시드
        
    Returns:
        처리 결과 정보
    """
    try:
        # 파일명에서 모델명 추출
        model_name = os.path.basename(file_path).replace('.json', '')
        similarity_method = os.path.basename(os.path.dirname(file_path))
        
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
        
        # CoT 정확도 계산을 위한 변수
        cot_correct_count = 0
        total_items = len(results)
        
        # 각 항목 기본 처리 (CoT 정답 추출 및 정확도 계산)
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
            
            # response_1에서 CoT 정답 추출
            cot_answer = None
            if "response_1" in processed_item:
                cot_answer = extract_numerical_answer(processed_item["response_1"])
                processed_item["cot_answer"] = cot_answer
            
            # CoT 답변이 정답과 일치하는지 확인
            if cot_answer is not None and correct_answer is not None:
                if is_numeric_answer_correct(cot_answer, correct_answer):
                    cot_correct_count += 1
                    processed_item["cot_is_correct"] = True
                else:
                    processed_item["cot_is_correct"] = False
            else:
                processed_item["cot_is_correct"] = False
            
            processed_results.append(processed_item)
        
        # CoT 정확도 계산
        cot_accuracy = cot_correct_count / total_items if total_items > 0 else 0
        
        # 1. 기존 self-consistency 적용
        sc_results = apply_original_self_consistency(processed_results, seed=seed)
        
        # 2. 여러 가중치 설정 기반 self-consistency 적용
        weighted_results = sc_results
        
        # 각 가중치 설정에 대해 처리
        weight_accuracies = {}
        
        for weights in weight_schemes:
            weight_info = '_'.join(map(str, weights))
            weighted_results = apply_weighted_self_consistency(weighted_results, weights, seed=seed)
            
            # 각 가중치 설정에 대한 정확도 계산
            correct_count = 0
            
            for item in weighted_results:
                weighted_sc_answer = item.get(f"weighted_sc_answer_{weight_info}")
                correct_answer = item.get("correct_answer")
                
                if weighted_sc_answer is not None and correct_answer is not None:
                    if is_numeric_answer_correct(weighted_sc_answer, correct_answer):
                        correct_count += 1
                        item[f"weighted_sc_is_correct_{weight_info}"] = True
                    else:
                        item[f"weighted_sc_is_correct_{weight_info}"] = False
                else:
                    item[f"weighted_sc_is_correct_{weight_info}"] = False
            
            # 해당 가중치 설정의 정확도 저장
            accuracy = correct_count / total_items if total_items > 0 else 0
            weight_accuracies[weight_info] = {
                "accuracy": accuracy
            }
        
        # 기존 self-consistency 정확도 계산
        sc_correct_count = 0
        
        for item in weighted_results:
            # 기존 self-consistency 정확도 계산
            sc_answer = item.get("self_consistency_answer")
            correct_answer = item.get("correct_answer")
            
            if sc_answer is not None and correct_answer is not None:
                if is_numeric_answer_correct(sc_answer, correct_answer):
                    sc_correct_count += 1
                    item["sc_is_correct"] = True
                else:
                    item["sc_is_correct"] = False
            else:
                item["sc_is_correct"] = False
        
        # 기존 Self-consistency 정확도 계산
        sc_accuracy = sc_correct_count / total_items if total_items > 0 else 0
        
        # 결과 반환
        final_result = {
            "model_name": model_name,
            "file_path": file_path,
            "similarity_method": similarity_method,
            "cot_accuracy": cot_accuracy,
            "sc_accuracy": sc_accuracy,
            "total_items": total_items,
            "weighted_results": weight_accuracies
        }
        
        return final_result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def run_multiple_experiments(
    file_paths: List[str], 
    weight_schemes: List[List[int]], 
    num_experiments: int = 10, 
    start_seed: int = 1
) -> Dict:
    """
    여러 번 실험을 수행하고 평균과 표준편차를 계산합니다.
    
    Args:
        file_paths: 처리할 파일 경로 목록
        weight_schemes: 여러 가중치 설정들의 리스트
        num_experiments: 실험 반복 횟수
        start_seed: 시작 시드
        
    Returns:
        실험 결과 정보
    """
    # 실험 결과 저장 구조
    all_experiments = {}
    
    # 각 파일에 대해
    for file_path in tqdm(file_paths, desc="Processing files"):
        file_name = os.path.basename(file_path)
        model_name = file_name.replace('.json', '')
        similarity_method = os.path.basename(os.path.dirname(file_path))
        
        # 실험 결과 저장
        model_experiments = {
            "seeds": [],
            "cot_accuracy": [],
            "sc_accuracy": [],
            "weighted_results": {w_info: [] for w_info in ['_'.join(map(str, w)) for w in weight_schemes]}
        }
        
        # 여러 시드로 실험 수행
        for exp_idx in range(num_experiments):
            seed = start_seed + exp_idx
            result = process_single_file(file_path, weight_schemes, seed)
            
            if result:
                model_experiments["seeds"].append(seed)
                model_experiments["cot_accuracy"].append(result["cot_accuracy"])
                model_experiments["sc_accuracy"].append(result["sc_accuracy"])
                
                for weight_info, accuracy_info in result["weighted_results"].items():
                    model_experiments["weighted_results"][weight_info].append(accuracy_info["accuracy"])
        
        # 실험 결과 저장
        all_experiments[f"{similarity_method}_{model_name}"] = model_experiments
    
    return all_experiments

def create_summary_with_std(all_experiments: Dict, weight_schemes: List[List[int]], output_dir: str):
    """
    모든 실험 결과의 평균과 표준편차를 CSV로 생성합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 가중치 정보 문자열 생성
    weight_headers = []
    for weights in weight_schemes:
        weight_info = '_'.join(map(str, weights))
        weight_headers.append(f"weighted_sc_{weight_info}")
    
    # CSV 파일 생성
    csv_path = os.path.join(output_dir, "multi_experiment_summary.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # CSV 헤더 정의
        fieldnames = [
            'similarity_method',
            'model_name', 
            'cot_accuracy', 
            'sc_accuracy'
        ] + weight_headers
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 각 모델/유사도 방법에 대한 결과 정리
        for key, experiments in all_experiments.items():
            parts = key.split('_', 1)
            similarity_method = parts[0]
            model_name = parts[1] if len(parts) > 1 else key
            
            # 평균 및 표준편차 계산
            cot_mean = np.mean(experiments["cot_accuracy"])
            cot_std = np.std(experiments["cot_accuracy"])
            
            sc_mean = np.mean(experiments["sc_accuracy"])
            sc_std = np.std(experiments["sc_accuracy"])
            
            # 결과 행 생성
            row = {
                'similarity_method': similarity_method,
                'model_name': model_name,
                'cot_accuracy': f"{cot_mean:.4f} ± {cot_std:.4f}",
                'sc_accuracy': f"{sc_mean:.4f} ± {sc_std:.4f}"
            }
            
            # 각 가중치 설정에 대한 평균 및 표준편차 추가
            for weights in weight_schemes:
                weight_info = '_'.join(map(str, weights))
                if weight_info in experiments["weighted_results"]:
                    w_mean = np.mean(experiments["weighted_results"][weight_info])
                    w_std = np.std(experiments["weighted_results"][weight_info])
                    row[f"weighted_sc_{weight_info}"] = f"{w_mean:.4f} ± {w_std:.4f}"
                else:
                    row[f"weighted_sc_{weight_info}"] = "N/A"
            
            writer.writerow(row)
    
    print(f"실험 결과 요약이 {csv_path}에 저장되었습니다.")
    return csv_path

def main():
    """
    메인 함수 - 여러 폴더의 모든 JSON 파일 처리 및 여러 시드로 실험
    """
    # 입력 폴더 목록
    INPUT_FOLDERS = [
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar",
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_F1_similar",
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_recall_similar",
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/cosine_similar",
        "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/precision_similar"
    ]
    
    # 출력 디렉토리
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/multi_seed_experiment"
    
    # 가중치 설정들 정의
    weight_schemes = [
        [5, 5, 4, 4, 3],  # 균형 잡힌 분포
        [5, 4, 4, 4, 2]   # 다른 분포
    ]
    
    # 실험 설정
    NUM_EXPERIMENTS = 10  # 10번 반복 실험
    START_SEED = 26    # 시작 시드
    
    # 커맨드 라인 인수로 디렉토리 지정 가능하게 함
    if len(sys.argv) > 1:
        OUTPUT_DIR = sys.argv[1]
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모든 파일 경로 수집
    all_file_paths = []
    for input_folder in INPUT_FOLDERS:
        # 모든 JSON 파일 목록 가져오기
        json_files = glob.glob(os.path.join(input_folder, "*.json"))
        all_file_paths.extend(json_files)
    
    # 파일이 없으면 종료
    if not all_file_paths:
        print("처리할 JSON 파일을 찾을 수 없습니다!")
        return
    
    print(f"총 {len(all_file_paths)}개 파일에 대해 각 {NUM_EXPERIMENTS}번씩 실험을 수행합니다.")
    
    # 여러 시드로 실험 수행
    all_experiments = run_multiple_experiments(
        all_file_paths, 
        weight_schemes, 
        num_experiments=NUM_EXPERIMENTS, 
        start_seed=START_SEED
    )
    
    # 결과 요약 CSV 생성
    summary_path = create_summary_with_std(all_experiments, weight_schemes, OUTPUT_DIR)
    
    print(f"\n모든 처리가 완료되었습니다.")
    print(f"실험 결과 요약은 {summary_path}에 저장되었습니다.")
    
if __name__ == "__main__":
    main()