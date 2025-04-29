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

def extract_answer(response: str) -> Optional[str]:
    """
    답변에서 정답 선택지(A, B, C, D, E)를 추출하는 함수
    """
    if not response:
        return None
    
    # CommonsenseQA 정답 패턴
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
            # 캡처된 그룹(선택지) 추출 및 대문자로 변환
            answer = match.group(1).strip().upper()
            return answer
    
    return None

def is_answer_correct(predicted: str, correct: str) -> bool:
    """
    예측된 답변이 정답과 일치하는지 확인하는 함수
    CommonsenseQA는 선택지 형태의 정답이므로 단순 문자열 비교를 수행
    """
    if predicted is None or correct is None:
        return False
    
    # 대소문자 구분 없이 비교
    return predicted.upper() == correct.upper()

def apply_weighted_self_consistency(results: List[Dict[str, Any]], weights: List[int]) -> List[Dict[str, Any]]:
    """
    매개변수화된 가중치를 사용하는 self-consistency 함수 - CommonsenseQA용
    
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
                        answer = extract_answer(response_val)
                        updated_item["weighted_sc_answer"] = answer
                        updated_item["weighted_sc_source"] = "single_response"
            else:
                # 아무것도 없으면 첫 번째 응답 사용
                updated_item["weighted_sc_method"] = "fallback_to_first"
                response_val = item.get("response_1")
                if isinstance(response_val, str):
                    answer = extract_answer(response_val)
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
                        answer = extract_answer(response_val)
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
                # 동률인 경우 첫 번째 항목 선택
                selected_answer = tied_answers[0]
                updated_item["weighted_sc_answer"] = selected_answer
                updated_item["weighted_sc_source"] = f"first_tie_breaker_from_{len(tied_answers)}"
                updated_item["weighted_sc_method"] = "weighted_vote_with_first_tie"
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
                answer = extract_answer(response_val)
                updated_item["weighted_sc_answer"] = answer
                updated_item["weighted_sc_source"] = "first_response"
        
        updated_results.append(updated_item)
    
    return updated_results

def evaluate_weighted_sc(results: List[Dict[str, Any]], weights: List[int]) -> Tuple[float, int, int]:
    """
    가중치 기반 self-consistency의 정확도를 평가합니다. - CommonsenseQA용
    
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
        
        # CommonsenseQA는 answerKey 또는 correct_answer 필드에 정답이 있음
        correct_answer = None
        if "answerKey" in item:
            correct_answer = item["answerKey"]
        elif "correct_answer" in item:
            correct_answer = item["correct_answer"]
            
        if weighted_sc_answer is not None and correct_answer is not None:
            weighted_sc_valid_count += 1
            if is_answer_correct(weighted_sc_answer, correct_answer):
                weighted_sc_correct_count += 1
    
    # 정확도 계산
    weighted_sc_accuracy = weighted_sc_correct_count / total_items if total_items > 0 else 0
    
    return weighted_sc_accuracy, weighted_sc_correct_count, total_items

def generate_descending_weights(max_weight=5):
    """
    내림차순 가중치 조합만 생성합니다.
    예: [5,4,3,2,1], [6,5,4,3,2], [5,5,4,3,1], ...
    
    Args:
        max_weight: 최대 가중치 값 (최소값은 0)
        
    Returns:
        내림차순 가중치 조합 리스트
    """
    descending_weights = []
    
    # 첫 번째 가중치부터 시작
    for w1 in range(max_weight, 0, -1):
        # 두 번째 가중치는 첫 번째 이하
        for w2 in range(w1, 0, -1):
            # 세 번째 가중치는 두 번째 이하
            for w3 in range(w2, 0, -1):
                # 네 번째 가중치는 세 번째 이하
                for w4 in range(w3, 0, -1):
                    # 다섯 번째 가중치는 네 번째 이하
                    for w5 in range(w4, 0, -1):
                        weights = [w1, w2, w3, w4, w5]
                        descending_weights.append(weights)
    
    return descending_weights

def run_descending_weights_experiment(results: List[Dict[str, Any]], output_dir: str, model_name: str, max_weight=6):
    """
    내림차순 가중치 조합만 실험합니다. - CommonsenseQA용
    
    Args:
        results: 원본 결과 데이터 리스트
        output_dir: 결과를 저장할 디렉토리
        model_name: 모델 이름
        max_weight: 최대 가중치 값
    """
    # 내림차순 가중치 조합 생성
    weight_combinations = generate_descending_weights(max_weight)
    
    print(f"총 {len(weight_combinations)}개 내림차순 가중치 조합을 테스트합니다.")
    
    # 결과를 저장할 리스트
    grid_search_results = []
    
    # 각 가중치 조합 평가
    for weights in tqdm(weight_combinations, desc="Testing descending weight combinations"):
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
    output_file = os.path.join(output_dir, f"descending_weight_search_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grid_search_results, f, ensure_ascii=False, indent=2)
    
    # CSV로도 저장
    csv_path = os.path.join(output_dir, f"descending_weight_search_{model_name}.csv")
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
    print("\n상위 10개 가중치 조합:")
    for i, result in enumerate(grid_search_results[:10], 1):
        print(f"{i}. {result['weights']} - 정확도: {float(result['accuracy']):.6f} ({result['correct_count']}/{result['total_items']})")
    
    return best_result

def extract_model_name(filename: str) -> str:
    """
    파일명에서 모델 이름을 추출하는 함수 - CommonsenseQA용
    """
    if filename.startswith("extracted_"):
        filename = filename[10:]
    
    # CommonsenseQA 파일명 패턴 추가
    patterns = [
        r'CommonsenseQA_([^_]+)_',
        r'CSQA_([^_]+)_',
        r'([^_]+)_CommonsenseQA',
        r'([^_]+)_CSQA',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return filename.replace('.json', '')

def main_descending_weights_experiment():
    """
    내림차순 가중치 실험을 위한 메인 함수 - CommonsenseQA용
    """
    # 상수 정의 - 적절한 경로로 수정
    INPUT_DIR = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/recall_similar"
    OUTPUT_DIR = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/recall_similar/descending_weight_experiments"
    
    # 커맨드 라인 인수로 디렉토리 지정 가능하게 함
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
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
    
    # 최대 가중치 값 설정
    max_weight = 5  # 0-5 범위
    
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
            print(f"경고: 파일 '{filename}'에서 'results' 키를 찾을 수 없습니다. 다음 파일로 넘어갑니다.")
            continue
        
        print(f"파일 '{filename}'에서 {len(results)}개 아이템을 로드했습니다.")
        
        # 내림차순 가중치 실험 수행
        print(f"모델 {model_name}에 대한 내림차순 가중치 실험 수행 중...")
        best_result = run_descending_weights_experiment(results, OUTPUT_DIR, model_name, max_weight)
        
        best_weights[model_name] = best_result
    
    # 모든 모델의 최적 가중치 저장
    summary_path = os.path.join(OUTPUT_DIR, "descending_best_weights_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(best_weights, f, ensure_ascii=False, indent=2)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"모델별 최적 가중치 요약은 {summary_path}에 저장되었습니다.")

def find_majority_better_weights(experiment_dir: str, threshold_percent: float = 75.0):
    """
    대부분의 CSV 파일에서 기준 가중치([1,1,1,1,1])보다 성능이 좋은 가중치 조합을 찾습니다.
    CommonsenseQA용
    
    Args:
        experiment_dir: 실험 결과가 저장된 디렉토리
        threshold_percent: 향상을 보여야 하는 모델의 최소 비율 (%)
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(experiment_dir, "descending_weight_search_*.csv"))
    
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
        model_name = os.path.basename(csv_file).replace("descending_weight_search_", "").replace(".csv", "")
        print(f"모델 {model_name} 분석 중...")
        
        # 가중치별 정확도 및 기준 가중치 정확도 찾기
        weights_accuracy = {}
        baseline_accuracy = None
        baseline_weights = [1, 1, 1, 1, 1]  # 모든 가중치가 1인 경우를 기준으로 사용
        
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
            # 모든 가중치가 1인 조합을 가장 가까운 조합으로 대체
            closest_weights = None
            min_diff = float('inf')
            for w in weights_accuracy.keys():
                # 기준 가중치와의 차이 계산
                diff = sum(abs(a - b) for a, b in zip(w, baseline_weights))
                if diff < min_diff:
                    min_diff = diff
                    closest_weights = w
            
            if closest_weights:
                baseline_accuracy = weights_accuracy[closest_weights]
                print(f"가장 가까운 가중치 {closest_weights}의 정확도를 기준값으로 사용합니다: {baseline_accuracy:.6f}")
            else:
                print(f"경고: 모델 {model_name}에 대한 기준값을 찾을 수 없습니다. 이 모델을 분석에서 제외합니다.")
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
            csv_file = os.path.join(experiment_dir, f"descending_weight_search_{model_name}.csv")
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
    print(f"\n=== {threshold_percent:.1f}% 이상의 모델에서 기준보다 나은 가중치 조합 ({len(sorted_weights)}개) ===")
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
    output_csv = os.path.join(experiment_dir, f"descending_majority_{int(threshold_percent)}percent_better_weights.csv")
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
    
    return [list(weights) for weights, _ in sorted_weights]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--find-majority":
        # 임계값 설정 (기본값 75%)
        threshold = 75.0
        if len(sys.argv) > 2:
            threshold = float(sys.argv[2])
        
        # 실험 디렉토리 경로
        experiment_dir = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_similar"
        if len(sys.argv) > 3:
            experiment_dir = sys.argv[3]
            
        print(f"임계값 {threshold}%로 대다수 모델에서 성능 향상을 보이는 가중치 조합 찾기...")
        find_majority_better_weights(experiment_dir, threshold)
    else:
        # 내림차순 가중치 실험 함수 실행
        main_descending_weights_experiment()