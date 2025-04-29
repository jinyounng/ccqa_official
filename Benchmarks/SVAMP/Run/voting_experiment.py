import os
import json
import re
import random
import sys
import csv
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
from tqdm import tqdm
import itertools

random.seed(48)
# 기존 extract_numerical_answer 함수
def extract_numerical_answer(response: str) -> Optional[str]:
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

def apply_weighted_self_consistency(results: List[Dict[str, Any]], weights: List[int]) -> List[Dict[str, Any]]:
    """
    매개변수화된 가중치를 사용하는 self-consistency 함수
    
    Args:
        results: 원본 결과 데이터 리스트
        weights: 각 순위(1~5)에 적용할 가중치 (예: [4, 4, 3, 3, 2])
        
    Returns:
        weighted self-consistency가 적용된 결과 데이터 리스트
    """
    updated_results = []
    
    for item in tqdm(results, desc=f"Applying weighted self-consistency with weights {weights}", leave=False):
        updated_item = item.copy()
        
        # most_similar_idxs가 있는지 확인
        most_similar_idxs = item.get("most_similar_idxs")
        if not most_similar_idxs or not isinstance(most_similar_idxs, list):
            # most_similar_idxs가 없으면 기존 most_similar_idx 사용
            most_similar_idx = item.get("most_similar_idx")
            if most_similar_idx is not None:
                updated_item["weighted_sc_method"] = "single_idx"
                response_key = f"response_{most_similar_idx}"
                if response_key in item:
                    # 해당 응답에서 정답 추출
                    response_val = item.get(response_key)
                    if isinstance(response_val, str):
                        answer = extract_numerical_answer(response_val)
                        updated_item["weighted_sc_answer"] = answer
                        updated_item["weighted_sc_source"] = "single_response"
            else:
                # 아무것도 없으면 첫 번째 응답 사용
                updated_item["weighted_sc_method"] = "fallback_to_first"
                response_val = item.get("response_1")
                if isinstance(response_val, str):
                    answer = extract_numerical_answer(response_val)
                    updated_item["weighted_sc_answer"] = answer
                    updated_item["weighted_sc_source"] = "first_response"
            
            updated_results.append(updated_item)
            continue
        
        # most_similar_idxs에서 상위 5개 인덱스 가져오기 (순서대로)
        top_indices = most_similar_idxs[:5]
        
        # 인덱스가 5개 미만이면 부족한 만큼 채움
        if len(top_indices) < 5:
            # 빠진 인덱스들 찾기 (1-5 중에서)
            missing_indices = [i for i in range(1, 6) if i not in top_indices]
            # 부족한 만큼 채우기
            top_indices.extend(missing_indices[:5-len(top_indices)])
        
        # 각 응답에서 정답 추출 및 가중치 적용
        answer_votes = []
        
        # 각 응답에 지정된 가중치만큼 투표 부여
        for idx, weight in enumerate(weights[:len(top_indices)]):
            if weight > 0:  # 0 가중치는 투표에서 제외
                response_key = f"response_{top_indices[idx]}"
                if response_key in item:
                    response_val = item.get(response_key)
                    if isinstance(response_val, str):
                        answer = extract_numerical_answer(response_val)
                        if answer:
                            # 가중치만큼 투표 부여
                            answer_votes.extend([answer] * weight)
        
        # 정답 투표 결과 저장
        updated_item["weighted_vote_indices"] = top_indices
        updated_item["weighted_vote_answers"] = answer_votes
        updated_item["weighted_vote_weights"] = weights[:len(top_indices)]  # 사용된 가중치 저장
        
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
                # 동률이 있으면 랜덤하게 선택
                item_id = item.get("id")
                question = item.get("question")
                if item_id is not None:
                    random.seed(hash(str(item_id)))
                elif question is not None and isinstance(question, str):
                    random.seed(hash(question))
                else:
                    random.seed(hash(str(answer_votes)))
                
                selected_answer = random.choice(tied_answers)
                updated_item["weighted_sc_answer"] = selected_answer
                updated_item["weighted_sc_source"] = f"random_tie_breaker_from_{len(tied_answers)}"
                updated_item["weighted_sc_method"] = "weighted_vote_with_random_tie"
            else:
                # 동률이 없으면 최다 득표 정답 선택
                updated_item["weighted_sc_answer"] = top_answer
                updated_item["weighted_sc_source"] = f"votes_{top_count}_of_{sum(weights[:len(top_indices)])}"
                updated_item["weighted_sc_method"] = "weighted_vote"
        else:
            # 투표 결과가 없으면 첫 번째 응답 사용
            updated_item["weighted_sc_method"] = "fallback_to_first"
            response_val = item.get("response_1")
            if isinstance(response_val, str):
                answer = extract_numerical_answer(response_val)
                updated_item["weighted_sc_answer"] = answer
                updated_item["weighted_sc_source"] = "first_response"
        
        updated_results.append(updated_item)
    
    return updated_results

def evaluate_weighted_sc(results: List[Dict[str, Any]], weights: List[int]) -> Tuple[float, int, int]:
    """
    가중치 기반 self-consistency의 정확도를 평가합니다.
    
    Args:
        results: 원본 결과 데이터 리스트
        weights: 적용할 가중치
        
    Returns:
        정확도, 정답 개수, 총 항목 수를 담은 튜플
    """
    # 가중치 적용
    weighted_results = apply_weighted_self_consistency(results, weights)
    
    total_items = len(weighted_results)
    weighted_sc_correct_count = 0
    weighted_sc_valid_count = 0
    
    # 정확도 계산
    for item in weighted_results:
        weighted_sc_answer = item.get("weighted_sc_answer")
        correct_answer = item.get("original_answer")
        
        if weighted_sc_answer is not None and correct_answer is not None:
            weighted_sc_valid_count += 1
            
            # 문자열로 변환하여 처리
            weighted_sc_str = str(weighted_sc_answer).strip()
            correct_str = str(correct_answer).strip()
            
            # 숫자만 추출하여 비교
            weighted_sc_numeric = re.sub(r'[^\d.]', '', weighted_sc_str)
            correct_numeric = re.sub(r'[^\d.]', '', correct_str)
            
            try:
                if weighted_sc_numeric and correct_numeric:
                    weighted_sc_float = float(weighted_sc_numeric)
                    correct_float = float(correct_numeric)
                    if abs(weighted_sc_float - correct_float) < 1e-5:
                        weighted_sc_correct_count += 1
                else:
                    # 숫자가 아닌 경우 문자열 비교
                    if weighted_sc_str == correct_str:
                        weighted_sc_correct_count += 1
            except ValueError:
                # 숫자로 변환할 수 없는 경우 단순 문자열 비교
                if weighted_sc_str == correct_str:
                    weighted_sc_correct_count += 1
    
    # 정확도 계산
    weighted_sc_accuracy = weighted_sc_correct_count / total_items if total_items > 0 else 0
    
    return weighted_sc_accuracy, weighted_sc_correct_count, total_items

def is_non_increasing(weights: List[int]) -> bool:
    """
    가중치 리스트가 내림차순(같거나 작아지는) 형태인지 확인합니다.
    예: [5,4,3,2,1] 또는 [5,5,4,4,3]은 True를 반환합니다.
    
    Args:
        weights: 확인할 가중치 리스트
        
    Returns:
        내림차순 여부(True/False)
    """
    for i in range(1, len(weights)):
        if weights[i] > weights[i-1]:
            return False
    return True

def generate_non_increasing_weight_combinations(max_weight: int = 5, length: int = 5) -> List[List[int]]:
    """
    내림차순 형태의 모든 가중치 조합을 생성합니다.
    
    Args:
        max_weight: 최대 가중치 값
        length: 가중치 리스트의 길이
        
    Returns:
        내림차순 가중치 조합 리스트
    """
    result = []
    
    def backtrack(current, start_idx, current_max):
        if len(current) == length:
            result.append(current[:])
            return
        
        for w in range(current_max, 0, -1):  # 현재 최대 가중치부터 1까지 내림차순
            current.append(w)
            backtrack(current, start_idx + 1, w)  # 다음 가중치는 현재 가중치보다 작거나 같아야 함
            current.pop()
    
    backtrack([], 0, max_weight)
    return result

def run_non_increasing_grid_search(results: List[Dict[str, Any]], output_dir: str, model_name: str, quick_test: bool = False):
    # 내림차순 가중치 조합 생성
    if quick_test:
        # 빠른 테스트를 위한 일부 내림차순 가중치 조합
        weight_combinations = [
            [5, 4, 3, 2, 1],
            [5, 5, 4, 3, 2],
            [5, 5, 5, 4, 3],
            [4, 3, 2, 1, 0],
            [5, 4, 3, 0, 0],
            [3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1]
        ]
    else:
        # 모든 내림차순 가중치 조합 생성 (최대 가중치 5)
        weight_combinations = generate_non_increasing_weight_combinations(max_weight=5, length=5)
    
    # 적어도 하나는 0보다 큰 값이 있어야 함
    weight_combinations = [comb for comb in weight_combinations if sum(comb) > 0]
    
    print(f"총 {len(weight_combinations)}개 내림차순 가중치 조합을 테스트합니다.")
    
    # 결과를 저장할 리스트
    grid_search_results = []
    
    # 각 가중치 조합 평가
    for weights in tqdm(weight_combinations, desc="Testing non-increasing weight combinations"):
        accuracy, correct_count, total_items = evaluate_weighted_sc(results, weights)
        
        grid_search_results.append({
            "weights": weights,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_items": total_items
        })
    
    # 정확도 기준으로 내림차순 정렬
    grid_search_results.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # 결과 저장
    output_file = os.path.join(output_dir, f"non_increasing_weight_search_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grid_search_results, f, ensure_ascii=False, indent=2)
    
    # CSV로도 저장
    csv_path = os.path.join(output_dir, f"non_increasing_weight_search_{model_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['weights', 'accuracy', 'correct_count', 'total_items']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in grid_search_results:
            writer.writerow({
                'weights': str(result['weights']),
                'accuracy': f"{float(result['accuracy']):.6f}",
                'correct_count': result['correct_count'],
                'total_items': result['total_items']
            })
    
    # 최고 정확도의 가중치 조합 출력
    best_result = grid_search_results[0]
    print(f"\n모델 {model_name}의 최적 내림차순 가중치 조합:")
    print(f"  가중치: {best_result['weights']}")
    print(f"  정확도: {float(best_result['accuracy']):.6f} ({best_result['correct_count']}/{best_result['total_items']})")
    
    # 상위 10개 결과 출력
    print("\n상위 10개 내림차순 가중치 조합:")
    for i, result in enumerate(grid_search_results[:10], 1):
        print(f"{i}. {result['weights']} - 정확도: {float(result['accuracy']):.6f} ({result['correct_count']}/{result['total_items']})")
    
    return best_result

def extract_model_name(filename: str) -> str:
    """
    파일명에서 모델 이름을 추출하는 함수
    """
    if filename.startswith("extracted_"):
        filename = filename[10:]
    
    patterns = [r'svamp_([^_]+)_',]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return "unknown-model"

def main_non_increasing_experiment():
    """
    내림차순 가중치 실험을 위한 메인 함수
    """
    # 상수 정의
    INPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar"
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar/weight_experiments"
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print(f"경고: {INPUT_DIR}에서 JSON 파일을 찾을 수 없습니다!")
        return
    
    print(f"총 {len(json_files)}개 파일을 처리합니다.")
    
    # 모델별 최적 가중치 저장
    best_weights = {}
    
    # 빠른 테스트를 위한 플래그
    quick_test = False  # True로 설정하면 제한된 가중치 조합만 테스트
    
    # 각 파일 처리
    for file_path in json_files:
        # 파일명에서 모델명 추출
        filename = os.path.basename(file_path)
        model_name = extract_model_name(filename)
        
        if not model_name:
            print(f"경고: 파일명 '{filename}'에서 모델 이름을 추출할 수 없습니다. 파일명을 그대로 사용합니다.")
            model_name = filename.replace('.json', '')
        
        print(f"\n파일 '{filename}' (모델: {model_name})을 처리합니다.")
        
        # 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 결과 항목 가져오기
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = data
        
        print(f"파일 '{filename}'에서 {len(results)}개 아이템을 로드했습니다.")
        
        # 내림차순 그리드 서치 수행
        print(f"모델 {model_name}에 대한 내림차순 그리드 서치 수행 중...")
        best_result = run_non_increasing_grid_search(results, OUTPUT_DIR, model_name, quick_test=quick_test)
        
        best_weights[model_name] = best_result
    
    # 모든 모델의 최적 가중치 저장
    summary_path = os.path.join(OUTPUT_DIR, "best_non_increasing_weights_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(best_weights, f, ensure_ascii=False, indent=2)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"모델별 최적 내림차순 가중치 요약은 {summary_path}에 저장되었습니다.")

def find_majority_better_non_increasing_weights(experiment_dir: str, threshold_percent: float = 75.0):
    """
    대부분의 CSV 파일에서 기준 가중치([1,1,1,1,1])보다 성능이 좋은 내림차순 가중치 조합을 찾습니다.
    
    Args:
        experiment_dir: 실험 결과가 저장된 디렉토리
        threshold_percent: 향상을 보여야 하는 모델의 최소 비율 (%)
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(experiment_dir, "non_increasing_weight_search_*.csv"))
    
    if not csv_files:
        print(f"경고: {experiment_dir}에서 CSV 파일을 찾을 수 없습니다!")
        return []
    
    print(f"총 {len(csv_files)}개 CSV 파일을 분석합니다.")
    
    # 임계값 계산 (모델 수 기준)
    threshold_count = max(1, int((threshold_percent / 100) * len(csv_files)))
    print(f"기준: 최소 {threshold_count}개/{len(csv_files)}개 모델({threshold_percent:.1f}%)에서 향상된 가중치 조합 찾기")
    
    # 모델별 기준 정확도
    baseline_accuracies = {}
    
    # 모델별 기준보다 나은 가중치 집합
    better_weights_by_model = {}
    
    # 가중치별 개선 모델 수 카운트
    weight_improvement_count = Counter()
    
    # 가중치 조합 전체 집합
    all_weights = set()
    
    # 각 CSV 파일 처리
    for csv_file in csv_files:
        # 파일명에서 모델명 추출
        model_name = os.path.basename(csv_file).replace("non_increasing_weight_search_", "").replace(".csv", "")
        print(f"모델 {model_name} 분석 중...")
        
        # 가중치별 정확도 및 기준 가중치 정확도 찾기
        weights_accuracy = {}
        baseline_accuracy = None
        baseline_weights = [1, 1, 1, 1, 1]
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 가중치 문자열을 리스트로 변환
                weights_str = row['weights'].strip('[]').split(',')
                weights = tuple(int(w.strip()) for w in weights_str)
                
                # 정확도 값 가져오기
                accuracy = float(row['accuracy'])
                
                # 모든 가중치 조합 추가
                all_weights.add(weights)
                
                # 가중치별 정확도 저장
                weights_accuracy[weights] = accuracy
                
                # 기준 가중치 정확도 기록
                if weights == tuple(baseline_weights):
                    baseline_accuracy = accuracy
        
        # 기준 가중치가 없는 경우 처리
        if baseline_accuracy is None:
            print(f"경고: 모델 {model_name}에서 기준 가중치 {baseline_weights}를 찾을 수 없습니다!")
            continue
        
        baseline_accuracies[model_name] = baseline_accuracy
        print(f"모델 {model_name}의 기준 가중치 정확도: {baseline_accuracy:.6f}")
        
        # 기준보다 나은 가중치 찾기
        better_weights = {weights for weights, acc in weights_accuracy.items() if acc > baseline_accuracy}
        better_weights_by_model[model_name] = better_weights
        
        # 각 가중치 조합이 이 모델에서 향상되었는지 카운트
        for weights in better_weights:
            weight_improvement_count[weights] += 1
        
        print(f"모델 {model_name}에서 기준보다 나은 가중치: {len(better_weights)}개")
    
    # 대다수 모델에서 향상된 가중치 찾기
    majority_better_weights = [weights for weights, count in weight_improvement_count.items() 
                              if count >= threshold_count]
    
    # 향상도 순으로 정렬하기 위한 정보 수집
    weight_performances = {}
    
    for weights in tqdm(majority_better_weights, desc="가중치 성능 분석 중"):
        performances = {}
        for model_name in better_weights_by_model.keys():
            # 해당 가중치 조합의 정확도 찾기
            csv_file = os.path.join(experiment_dir, f"non_increasing_weight_search_{model_name}.csv")
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    weights_str = row['weights'].strip('[]').split(',')
                    row_weights = tuple(int(w.strip()) for w in weights_str)
                    
                    if row_weights == weights:
                        accuracy = float(row['accuracy'])
                        baseline = baseline_accuracies[model_name]
                        improvement = accuracy - baseline
                        performances[model_name] = (accuracy, improvement)
                        break
        
        # 총합 및 평균 개선도 계산 (모든 모델 대상)
        total_improvement = sum(imp for _, imp in performances.values())
        avg_improvement = total_improvement / len(performances) if performances else 0
        
        # 향상된 모델 수
        improved_count = weight_improvement_count[weights]
        improved_percent = (improved_count / len(better_weights_by_model)) * 100
        
        weight_performances[weights] = {
            'model_performances': performances,
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'improved_count': improved_count,
            'improved_percent': improved_percent
        }
    
    # 평균 개선도 기준으로 정렬
    sorted_weights = sorted(
        weight_performances.items(), 
        key=lambda x: x[1]['avg_improvement'], 
        reverse=True
    )
    
    # 결과 출력
    print(f"\n=== {threshold_percent:.1f}% 이상의 모델에서 기준보다 나은 내림차순 가중치 조합 ({len(sorted_weights)}개) ===")
    if not sorted_weights:
        print("조건을 만족하는 가중치 조합이 없습니다.")
        return []
    
    print("(평균 개선도 기준 내림차순 정렬)")
    print("=" * 80)
    print(f"{'가중치 조합':<20} {'향상 모델':<15} {'평균 개선도':<15} {'총 개선도':<15} {'모델별 변화'}")
    print("-" * 80)
    
    for weights, perf in sorted_weights[:20]:  # 상위 20개만 출력
        weights_str = str(list(weights))
        improved = f"{perf['improved_count']}/{len(better_weights_by_model)} ({perf['improved_percent']:.1f}%)"
        avg_imp = perf['avg_improvement']
        total_imp = perf['total_improvement']
        
        print(f"{weights_str:<20} {improved:<15} {avg_imp:.6f}   {total_imp:.6f}   ", end="")
        
        # 모델별 개선도 출력 (간결하게)
        model_perfs = []
        for model, (acc, imp) in perf['model_performances'].items():
            # 개선 여부에 따라 +/- 기호 추가
            sign = '+' if imp > 0 else ''
            model_perfs.append(f"{model}:{sign}{imp:.6f}")
        
        print(", ".join(model_perfs[:3]) + (", ..." if len(model_perfs) > 3 else ""))
    
    print("=" * 80)
    
    # 결과를 CSV로 저장
    output_csv = os.path.join(experiment_dir, f"majority_{int(threshold_percent)}percent_better_non_increasing_weights.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'avg_improvement', 'total_improvement'] + list(better_weights_by_model.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for weights, perf in sorted_weights:
            row = {
                'weights': str(list(weights)),
                'improved_count': perf['improved_count'],
                'improved_percent': perf['improved_percent'],
                'avg_improvement': perf['avg_improvement'],
                'total_improvement': perf['total_improvement']
            }
            
            # 모델별 개선도 추가
            for model, (acc, imp) in perf['model_performances'].items():
                row[model] = imp
            
            writer.writerow(row)
    
    print(f"\n결과가 {output_csv}에 저장되었습니다.")
    
    # 상세 정보 CSV 저장
    detailed_csv = os.path.join(experiment_dir, f"majority_{int(threshold_percent)}percent_non_increasing_details.csv")
    with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
        models = list(better_weights_by_model.keys())
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'avg_improvement']
        for model in models:
            fieldnames.append(f"{model}_accuracy")
            fieldnames.append(f"{model}_improvement")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for weights, perf in sorted_weights:
            row = {
                'weights': str(list(weights)),
                'improved_count': perf['improved_count'],
                'improved_percent': perf['improved_percent'],
                'avg_improvement': perf['avg_improvement']
            }
            
            # 모델별 정확도와 개선도 추가
            for model, (acc, imp) in perf['model_performances'].items():
                row[f"{model}_accuracy"] = acc
                row[f"{model}_improvement"] = imp
            
            writer.writerow(row)
    
    print(f"상세 정보가 {detailed_csv}에 저장되었습니다.")
    
    
def main():
    
    print("=" * 80)
    print("내림차순 가중치 최적화 실험 시작")
    print("=" * 80)
    
    # 실험 결과 디렉토리 지정
    INPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar"
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/roberta_precision_similar/weight_experiments"
    
    # 새로운 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"입력 디렉토리: {INPUT_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    
    # 1. 내림차순 가중치 그리드 서치 수행
    print("\n1. 내림차순 가중치 그리드 서치 수행 중...")
    main_non_increasing_experiment()
    
    # 2. 다수 모델에서 좋은 성능을 보인 내림차순 가중치 찾기
    print("\n2. 다수 모델에서 효과적인 내림차순 가중치 찾는 중...")
    threshold_percent = 75.0  # 75% 이상 모델에서 향상을 보이는 가중치 찾기
    majority_weights = find_majority_better_non_increasing_weights(OUTPUT_DIR, threshold_percent)
    
    # 추천 가중치 (상위 5개)
    if majority_weights:
        print("\n추천 내림차순 가중치 조합 (상위 5개):")
        for i, weights in enumerate(majority_weights[:5], 1):
            print(f"{i}. {weights}")
    
    # 임계값을 점진적으로 낮추면서 다시 시도
    if not majority_weights:
        for new_threshold in [70.0, 60.0, 50.0]:
            print(f"\n\n임계값을 {new_threshold}%로 낮추어 다시 시도합니다.")
            majority_weights = find_majority_better_non_increasing_weights(OUTPUT_DIR, new_threshold)
            if majority_weights:
                print(f"\n{new_threshold}% 임계값으로 찾은 추천 내림차순 가중치 조합 (상위 5개):")
                for i, weights in enumerate(majority_weights[:5], 1):
                    print(f"{i}. {weights}")
                break
    
    print("\n" + "=" * 80)
    print("내림차순 가중치 최적화 실험 완료")
    print("=" * 80)

if __name__ == "__main__":
    main()