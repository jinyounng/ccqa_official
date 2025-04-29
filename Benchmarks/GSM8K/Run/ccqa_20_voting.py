import os
import json
import re
import random
import sys
import csv
import glob
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm

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

def apply_original_self_consistency(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    기존 self-consistency 로직을 구현한 함수 - 동률 시 첫 번째 응답을 선택하도록 수정
    20개 생성을 사용하도록 수정됨
    
    Args:
        results: 원본 결과 데이터 리스트
        
    Returns:
        self-consistency가 적용된 결과 데이터 리스트
    """
    updated_results = []
    
    for item in tqdm(results, desc="Applying original self-consistency (20 indices)", leave=False):
        updated_item = item.copy()
        
        # 모든 response_n 키 찾기 (최대 20개)
        responses = []
        for i in range(1, 21):  # 최대 20개까지 검사
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
                # 동률인 경우 첫 번째 응답을 선택 (랜덤 선택 대신)
                updated_item["self_consistency_answer"] = top_answers[0]  # 리스트의 첫 번째 항목 선택
                updated_item["self_consistency_tie_breaker"] = "first_of_tied"  # 동률 처리 방법 기록
            
            # 투표 결과 저장
            updated_item["self_consistency_vote"] = {ans: count for ans, count in most_common_answers}
        
        updated_results.append(updated_item)
    
    return updated_results

def apply_weighted_self_consistency(results: List[Dict[str, Any]], weights: List[int]) -> List[Dict[str, Any]]:
    """
    가중치 기반 self-consistency 적용 (20개 인덱스 사용)
    
    Args:
        results: 원본 결과 데이터 리스트
        weights: 상위 응답들에 적용할 가중치 리스트 (최대 20개의 가중치 사용)
        
    Returns:
        가중치 기반 self-consistency가 적용된 결과 데이터 리스트
    """
    updated_results = []
    
    # 가중치 정보를 문자열로 변환하여 필드에 저장
    weight_info = '_'.join(map(str, weights))
    
    for item in tqdm(results, desc=f"Applying weighted SC [{weight_info}] (20 indices)", leave=False):
        updated_item = item.copy()
        
        # most_similar_idxs가 있는지 확인
        if "most_similar_idxs" not in item or not item["most_similar_idxs"]:
            # most_similar_idxs가 없으면 기존 most_similar_idx 사용
            if "most_similar_idx" in item:
                updated_item["weighted_sc_method"] = "single_idx"
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
        
        # most_similar_idxs에서 상위 20개 인덱스만 가져오기 (순서대로)
        top_indices = item["most_similar_idxs"][:20]  # 상위 20개만 사용
        
        # 인덱스가 20개보다 적으면 부족한 만큼 채움
        if len(top_indices) < 20:
            # 빠진 인덱스들 찾기 (1-20 중에서)
            missing_indices = [i for i in range(1, 21) if i not in top_indices]
            # 부족한 만큼 채우기
            top_indices.extend(missing_indices[:20-len(top_indices)])
        
        # 사용할 인덱스 20개와 가중치 적용
        use_indices = top_indices[:20]
        # 가중치의 길이 조정 (weights 리스트의 길이가 20보다 작을 경우 마지막 가중치로 채우기)
        use_weights = weights.copy()
        if len(use_weights) < 20:
            last_weight = use_weights[-1] if use_weights else 1
            use_weights.extend([last_weight] * (20 - len(use_weights)))
        else:
            use_weights = use_weights[:20]
        
        # 각 응답에서 정답 추출
        answer_votes = []
        extracted_answers = []
        
        # 상위 20개 응답 각각에 가중치 적용
        for idx, response_idx in enumerate(use_indices):
            response_key = f"response_{response_idx}"
            if response_key in item and item[response_key]:
                answer = extract_numerical_answer(item[response_key])
                extracted_answers.append(answer)
                if answer:
                    # 해당 인덱스의 가중치만큼 표 부여
                    weight = use_weights[idx]
                    answer_votes.extend([answer] * weight)
        
        # 정답 투표 결과 저장
        updated_item[f"weighted_vote_indices_{weight_info}"] = use_indices
        updated_item[f"weighted_vote_answers_{weight_info}"] = answer_votes
        updated_item[f"weighted_vote_weights_{weight_info}"] = use_weights  # 사용된 가중치 체계 저장
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
                # 동률인 경우 첫 번째 응답을 선택 
                selected_answer = tied_answers[0]  # 리스트의 첫 번째 항목 선택
                updated_item[f"weighted_sc_answer_{weight_info}"] = selected_answer
                updated_item[f"weighted_sc_source_{weight_info}"] = f"first_of_tied_{len(tied_answers)}"
                updated_item[f"weighted_sc_method_{weight_info}"] = "advanced_weighted_vote_with_tie"
                updated_item[f"weighted_sc_tie_breaker_{weight_info}"] = "first_of_tied"  # 동률 처리 방법 기록
            else:
                # 동률이 없으면 최다 득표 정답 선택
                updated_item[f"weighted_sc_answer_{weight_info}"] = top_answer
                updated_item[f"weighted_sc_source_{weight_info}"] = f"votes_{top_count}_of_{sum(vote_counts.values())}"
                updated_item[f"weighted_sc_method_{weight_info}"] = "advanced_weighted_vote"
        else:
            # 투표 결과가 없으면 첫 번째 응답 사용
            updated_item[f"weighted_sc_method_{weight_info}"] = "fallback_to_first"
            if "response_1" in item:
                answer = extract_numerical_answer(item["response_1"])
                updated_item[f"weighted_sc_answer_{weight_info}"] = answer
                updated_item[f"weighted_sc_source_{weight_info}"] = "first_response"
        
        updated_results.append(updated_item)
    
    return updated_results

def process_file(file_path: str, weight_schemes: List[List[int]]) -> Dict:
    """
    파일을 처리하여 CoT, Self-Consistency, 여러 가중치 기반 Self-Consistency를 적용합니다.
    모든 과정에서 20개 인덱스를 사용하도록 수정됨
    
    Args:
        file_path: 입력 파일 경로
        weight_schemes: 여러 가중치 설정들의 리스트
        
    Returns:
        처리 결과 정보
    """
    try:
        # 파일명에서 모델명 추출
        filename = os.path.basename(file_path)
        model_name = extract_model_name(filename)
        
        if not model_name:
            model_name = filename.replace('.json', '')
        
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
        
        # 처리용 결과 복사
        processed_results = []
        
        # CoT 정확도 계산을 위한 변수
        cot_correct_count = 0
        cot_valid_count = 0
        total_items = len(results)
        
        # 각 항목 기본 처리 (CoT 정답 추출 및 정확도 계산)
        for item in results:
            processed_item = item.copy()
            
            # 정답 필드 확인
            if "correct_answer" not in processed_item and "original_answer" in processed_item:
                processed_item["correct_answer"] = processed_item["original_answer"]
            
            # response_1에서 CoT 정답 추출
            cot_answer = None
            if "response_1" in processed_item:
                cot_answer = extract_numerical_answer(processed_item["response_1"])
                processed_item["cot_answer"] = cot_answer
            
            # CoT 답변이 정답과 일치하는지 확인
            if cot_answer is not None and "correct_answer" in processed_item:
                cot_valid_count += 1
                correct_answer = processed_item["correct_answer"]
                
                # 숫자만 추출하여 비교
                cot_numeric = re.sub(r'[^\d.]', '', str(cot_answer).strip())
                correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
                
                try:
                    if cot_numeric and correct_numeric:
                        cot_float = float(cot_numeric)
                        correct_float = float(correct_numeric)
                        if abs(cot_float - correct_float) < 1e-5:
                            cot_correct_count += 1
                            processed_item["cot_is_correct"] = True
                        else:
                            processed_item["cot_is_correct"] = False
                    else:
                        # 숫자가 아닌 경우 문자열 비교
                        if str(cot_answer).strip() == str(correct_answer).strip():
                            cot_correct_count += 1
                            processed_item["cot_is_correct"] = True
                        else:
                            processed_item["cot_is_correct"] = False
                except ValueError:
                    # 숫자로 변환할 수 없는 경우 단순 문자열 비교
                    if str(cot_answer).strip() == str(correct_answer).strip():
                        cot_correct_count += 1
                        processed_item["cot_is_correct"] = True
                    else:
                        processed_item["cot_is_correct"] = False
            else:
                processed_item["cot_is_correct"] = False
            
            processed_results.append(processed_item)
        
        # CoT 정확도 계산
        cot_accuracy = cot_correct_count / total_items if total_items > 0 else 0
        
        # 1. 기존 self-consistency 적용 (20개 사용)
        sc_results = apply_original_self_consistency(processed_results)
        
        # 2. 여러 가중치 설정 기반 self-consistency 적용 (20개 사용)
        weighted_results = sc_results
        
        # 각 가중치 설정에 대해 처리
        weight_accuracies = {}
        
        for weights in weight_schemes:
            # 가중치 정보 문자열
            weight_info = '_'.join(map(str, weights))
            
            weighted_results = apply_weighted_self_consistency(weighted_results, weights)
            
            # 각 가중치 설정에 대한 정확도 계산
            correct_count = 0
            valid_count = 0
            
            for item in weighted_results:
                weighted_sc_answer = item.get(f"weighted_sc_answer_{weight_info}")
                correct_answer = item.get("correct_answer")
                
                if weighted_sc_answer is not None and correct_answer is not None:
                    valid_count += 1
                    
                    # 숫자만 추출하여 비교
                    weighted_sc_numeric = re.sub(r'[^\d.]', '', str(weighted_sc_answer).strip())
                    correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
                    
                    try:
                        if weighted_sc_numeric and correct_numeric:
                            weighted_sc_float = float(weighted_sc_numeric)
                            correct_float = float(correct_numeric)
                            if abs(weighted_sc_float - correct_float) < 1e-5:
                                correct_count += 1
                                item[f"weighted_sc_is_correct_{weight_info}"] = True
                            else:
                                item[f"weighted_sc_is_correct_{weight_info}"] = False
                        else:
                            # 숫자가 아닌 경우 문자열 비교
                            if str(weighted_sc_answer).strip() == str(correct_answer).strip():
                                correct_count += 1
                                item[f"weighted_sc_is_correct_{weight_info}"] = True
                            else:
                                item[f"weighted_sc_is_correct_{weight_info}"] = False
                    except ValueError:
                        # 숫자로 변환할 수 없는 경우 단순 문자열 비교
                        if str(weighted_sc_answer).strip() == str(correct_answer).strip():
                            correct_count += 1
                            item[f"weighted_sc_is_correct_{weight_info}"] = True
                        else:
                            item[f"weighted_sc_is_correct_{weight_info}"] = False
                else:
                    item[f"weighted_sc_is_correct_{weight_info}"] = False
            
            # 해당 가중치 설정의 정확도 저장
            accuracy = correct_count / total_items if total_items > 0 else 0
            weight_accuracies[weight_info] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "valid_count": valid_count
            }
        
        # 기존 self-consistency 정확도 계산
        sc_correct_count = 0
        sc_valid_count = 0
        
        for item in weighted_results:
            # 기존 self-consistency 정확도 계산
            sc_answer = item.get("self_consistency_answer")
            correct_answer = item.get("correct_answer")
            
            if sc_answer is not None and correct_answer is not None:
                sc_valid_count += 1
                
                # 숫자만 추출하여 비교
                sc_numeric = re.sub(r'[^\d.]', '', str(sc_answer).strip())
                correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
                
                try:
                    if sc_numeric and correct_numeric:
                        sc_float = float(sc_numeric)
                        correct_float = float(correct_numeric)
                        if abs(sc_float - correct_float) < 1e-5:
                            sc_correct_count += 1
                            item["sc_is_correct"] = True
                        else:
                            item["sc_is_correct"] = False
                    else:
                        # 숫자가 아닌 경우 문자열 비교
                        if str(sc_answer).strip() == str(correct_answer).strip():
                            sc_correct_count += 1
                            item["sc_is_correct"] = True
                        else:
                            item["sc_is_correct"] = False
                except ValueError:
                    # 숫자로 변환할 수 없는 경우 단순 문자열 비교
                    if str(sc_answer).strip() == str(correct_answer).strip():
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
            "cot_accuracy": cot_accuracy,
            "cot_correct_count": cot_correct_count,
            "cot_valid_count": cot_valid_count,
            "sc_accuracy": sc_accuracy,
            "sc_correct_count": sc_correct_count,
            "sc_valid_count": sc_valid_count,
            "total_items": total_items,
            "weighted_results": weight_accuracies
        }
        
        return final_result
    
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_model_name(filename: str) -> str:
    """
    파일명에서 모델 이름을 추출하는 함수
    """
    if filename.startswith("extracted_"):
        filename = filename[10:]
    
    patterns = [r'GSM8K_([^_]+)_',]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return "unknown-model"

def extract_similarity_method(folder_path: str) -> str:
    """
    폴더 경로에서 유사도 방법 추출
    """
    # 폴더 이름에서 유사도 방법 추출 (예: roberta_precision_similar -> roberta_precision)
    folder_name = os.path.basename(folder_path.rstrip('/'))
    if '_similar' in folder_name:
        return folder_name.replace('_similar', '')
    return folder_name

def main():
    """
    메인 함수 - 여러 폴더의 모든 JSON 파일 처리
    20개 인덱스를 사용하도록 수정됨
    """
    # 입력 폴더 목록
    INPUT_FOLDERS = [
        "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_3generation_result/roberta_precision_similar"
    ]
    
    # 출력 디렉토리
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_3generation_result/combined_sc_comparison"
    
    # 가중치 설정들 정의 (더 많은 인덱스를 위한 가중치)
    weight_schemes = [
        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # 선형 감소
        [5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # 단계적 감소
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 상위 10개만 감소
        [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # 단계적 감소 (더 완만)
        [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1]    # 더 높은 가중치
    ]
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모든 폴더의 모든 JSON 파일 처리 결과 저장
    all_results = []
    
    # 폴더별 처리
    for input_folder in INPUT_FOLDERS:
        # 유사도 방법 추출 
        similarity_method = extract_similarity_method(input_folder)
        
        print(f"\n폴더 '{similarity_method}' 처리 중...")
        
        # 모든 JSON 파일 목록 가져오기
        json_files = glob.glob(os.path.join(input_folder, "*.json"))
        
        if not json_files:
            print(f"경고: {input_folder}에서 JSON 파일을 찾을 수 없습니다!")
            continue
        
        print(f"총 {len(json_files)}개 파일을 처리합니다.")
        
        # 각 파일 처리
        for file_path in tqdm(json_files, desc=f"{similarity_method} 파일 처리 중"):
            result = process_file(file_path, weight_schemes)
            
            if result:
                # 유사도 방법 추가
                result["similarity_method"] = similarity_method
                all_results.append(result)
    
    # 결과 통합 및 CSV 생성
    create_comparison_csv(all_results, weight_schemes, OUTPUT_DIR)

def create_comparison_csv(all_results: List[Dict], weight_schemes: List[List[int]], output_dir: str):
    """
    모든 폴더, 모델에 대한 비교 결과를 CSV로 생성
    """
    # 결과 요약 CSV 생성
    csv_path = os.path.join(output_dir, "multi_folder_sc_comparison.csv")
    
    # 가중치 정보 문자열 생성
    weight_headers = []
    for weights in weight_schemes:
        weight_info = '_'.join(map(str, weights))
        weight_headers.append(f"weighted_sc_{weight_info}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # CSV 헤더 정의
        fieldnames = [
            'similarity_method',
            'model_name', 
            'cot_accuracy', 
            'sc_accuracy'
        ] + weight_headers + ['total_items']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 각 결과 정보 작성
        for result in all_results:
            row = {
                'similarity_method': result['similarity_method'],
                'model_name': result['model_name'],
                'cot_accuracy': f"{result['cot_accuracy']:.4f}",
                'sc_accuracy': f"{result['sc_accuracy']:.4f}",
                'total_items': result['total_items']
            }
            
            # 각 가중치 설정의 정확도 추가
            for weights in weight_schemes:
                weight_info = '_'.join(map(str, weights))
                if weight_info in result['weighted_results']:
                    accuracy = result['weighted_results'][weight_info]['accuracy']
                    row[f"weighted_sc_{weight_info}"] = f"{accuracy:.4f}"
                else:
                    row[f"weighted_sc_{weight_info}"] = "N/A"
            
            writer.writerow(row)
    
    # 가중치별 최상위 성능 모델 찾기
    best_by_weight = {}
    
    for weights in weight_schemes:
        weight_info = '_'.join(map(str, weights))
        
        # 각 유사도 방법 및 모델 조합에 대한 최상위 성능 찾기
        best_by_similarity = {}
        
        for result in all_results:
            sim_method = result['similarity_method']
            model_name = result['model_name']
            
            if weight_info in result['weighted_results']:
                accuracy = result['weighted_results'][weight_info]['accuracy']
                
                key = sim_method
                if key not in best_by_similarity or accuracy > best_by_similarity[key]['accuracy']:
                    best_by_similarity[key] = {
                        'model_name': model_name,
                        'accuracy': accuracy
                    }
        
        best_by_weight[weight_info] = best_by_similarity
    
    # 최상위 성능 정보 CSV 생성
    best_csv_path = os.path.join(output_dir, "best_models_by_weight.csv")
    
    with open(best_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        writer.writerow(['Weight Scheme', 'Similarity Method', 'Best Model', 'Accuracy'])
        
        # 각 가중치 설정별로 최상위 모델 작성
        for weights in weight_schemes:
            weight_info = '_'.join(map(str, weights))
            # 가중치가 많아 CSV가 읽기 어려울 수 있으므로 간단한 설명으로 대체
            if len(weights) > 10:
                weight_desc = f"{weights[0]}-{weights[-1]} ({len(weights)}개)"
            else:
                weight_desc = f"[{', '.join(map(str, weights[:10]))}{'...' if len(weights) > 10 else ''}]"
                
            writer.writerow([weight_desc, "", "", ""])
            
            for sim_method, best in best_by_weight[weight_info].items():
                writer.writerow(["", sim_method, best['model_name'], f"{best['accuracy']:.4f}"])
            
            writer.writerow(["", "", "", ""])  # 빈 행 추가
    
    # 요약 테이블: 가중치별 평균 성능
    summary_csv_path = os.path.join(output_dir, "weight_scheme_summary.csv")
    
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        writer.writerow(['Weight Scheme', 'Avg Accuracy Across All Models', 'Avg Accuracy Best Model per Sim Method', 'Best Overall Model', 'Best Accuracy'])
        
        # 각 가중치 설정에 대한 요약 통계 계산
        for weights in weight_schemes:
            weight_info = '_'.join(map(str, weights))
            
            # 가중치가 많아 CSV가 읽기 어려울 수 있으므로 간단한 설명으로 대체
            if len(weights) > 10:
                weight_desc = f"{weights[0]}-{weights[-1]} ({len(weights)}개)"
            else:
                weight_desc = f"[{', '.join(map(str, weights[:10]))}{'...' if len(weights) > 10 else ''}]"
            
            # 모든 모델에 대한 평균 성능
            all_accuracies = []
            for result in all_results:
                if weight_info in result['weighted_results']:
                    all_accuracies.append(result['weighted_results'][weight_info]['accuracy'])
            
            avg_all = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
            
            # 각 유사도 방법별 최상위 모델의 평균 성능
            best_accuracies = [best['accuracy'] for best in best_by_weight[weight_info].values()]
            avg_best = sum(best_accuracies) / len(best_accuracies) if best_accuracies else 0
            
            # 전체 최상위 모델 찾기
            best_model = None
            best_accuracy = 0
            best_sim = None
            
            for sim_method, best in best_by_weight[weight_info].items():
                if best['accuracy'] > best_accuracy:
                    best_accuracy = best['accuracy']
                    best_model = best['model_name']
                    best_sim = sim_method
            
            writer.writerow([
                weight_desc,
                f"{avg_all:.4f}",
                f"{avg_best:.4f}",
                f"{best_sim} - {best_model}",
                f"{best_accuracy:.4f}"
            ])
    
    # 유사도 방법별 요약
    similarity_csv_path = os.path.join(output_dir, "similarity_method_summary.csv")
    
    with open(similarity_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        header = ['Similarity Method', 'CoT Avg', 'SC Avg']
        for weights in weight_schemes:
            # 가중치가 많아 CSV가 읽기 어려울 수 있으므로 간단한 설명으로 대체
            if len(weights) > 10:
                weight_desc = f"WSC [{weights[0]}-{weights[-1]}]"
            else:
                weight_desc = f"WSC [{', '.join(map(str, weights[:3]))}{'...' if len(weights) > 3 else ''}]"
            header.append(weight_desc)
        
        writer.writerow(header)
        
        # 각 유사도 방법별 요약 통계
        similarity_methods = set(result['similarity_method'] for result in all_results)
        
        for sim_method in similarity_methods:
            # 해당 유사도 방법의 모든 결과 가져오기
            sim_results = [r for r in all_results if r['similarity_method'] == sim_method]
            
            if not sim_results:
                continue
            
            # CoT 및 SC 평균 정확도 계산
            cot_avg = sum(r['cot_accuracy'] for r in sim_results) / len(sim_results)
            sc_avg = sum(r['sc_accuracy'] for r in sim_results) / len(sim_results)
            
            row = [sim_method, f"{cot_avg:.4f}", f"{sc_avg:.4f}"]
            
            # 각 가중치 설정의 평균 정확도 계산
            for weights in weight_schemes:
                weight_info = '_'.join(map(str, weights))
                accuracies = []
                
                for result in sim_results:
                    if weight_info in result['weighted_results']:
                        accuracies.append(result['weighted_results'][weight_info]['accuracy'])
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    row.append(f"{avg_accuracy:.4f}")
                else:
                    row.append("N/A")
            
            writer.writerow(row)
    
    print(f"\n모든 처리가 완료되었습니다.")
    print(f"비교 결과는 {csv_path}에 저장되었습니다.")
    print(f"가중치별 최상위 모델 정보는 {best_csv_path}에 저장되었습니다.")
    print(f"가중치 요약 정보는 {summary_csv_path}에 저장되었습니다.")
    print(f"유사도 방법 요약 정보는 {similarity_csv_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()