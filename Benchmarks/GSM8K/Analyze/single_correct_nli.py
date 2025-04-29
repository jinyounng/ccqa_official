import os
import json
import re
import numpy as np
import glob
from collections import Counter
from tqdm import tqdm
import pandas as pd
from os import path
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

def analyze_single_correct_problems(results):
    """
    정답이 하나인 문제의 특성을 분석하는 함수 - precision 점수 기반으로 수정
    """
    single_correct_features = []
    
    for item in results:
        correct_answer = item.get("correct_answer")
        if not correct_answer:
            continue
            
        # 각 응답에서 정답 추출
        correct_counts = 0
        extracted_answers = []
        correct_indices = []
        
        for i in range(1, 6):
            response_key = f"response_{i}"
            if response_key in item and item[response_key]:
                answer = extract_numerical_answer(item[response_key])
                extracted_answers.append(answer)
                
                # 정답과 일치하는지 확인
                if is_answer_correct(answer, correct_answer):
                    correct_counts += 1
                    correct_indices.append(i)
        
        # 정답이 하나인 문제만 분석
        if correct_counts == 1:
            # 모든 유효한 답변 추출 (None이 아닌 것만)
            valid_answers = [a for a in extracted_answers if a]
            
            # 정답 인덱스 찾기
            correct_index = correct_indices[0] - 1  # 0-based 인덱스로 변환
            
            # 오답들만 모으기 (정답 인덱스를 제외)
            incorrect_answers = valid_answers.copy()
            if 0 <= correct_index < len(incorrect_answers):
                correct_value = incorrect_answers[correct_index]
                incorrect_answers.remove(correct_value)
            
            # 오답들 중 가장 많이 나온 것 찾기
            if incorrect_answers:
                incorrect_counter = Counter(incorrect_answers)
                most_common_wrong = incorrect_counter.most_common(1)[0]
                most_common_wrong_answer = most_common_wrong[0]
                most_common_wrong_count = most_common_wrong[1]
            else:
                most_common_wrong_answer = None
                most_common_wrong_count = 0
            
            # Precision 점수 분석 (entailment_min 대신 precision 사용)
            precision_scores = []
            correct_precision_score = 0
            max_precision_score = 0
            max_precision_index = 0
            
            for i in range(1, 6):
                if "similarity_scores" in item and f"question_{i}" in item["similarity_scores"]:
                    # precision 점수 가져오기 (없으면 0)
                    score = item["similarity_scores"][f"question_{i}"].get("precision", 0)
                    precision_scores.append(score)
                    
                    if i == correct_indices[0]:  # 정답 인덱스와 비교 (1-based)
                        correct_precision_score = score
                        
                    if score > max_precision_score:
                        max_precision_score = score
                        max_precision_index = i
            
            # 결과 저장
            feature = {
                "question_id": item.get("id", "unknown"),
                "correct_answer": correct_answer,
                "correct_index": correct_indices[0],  # 1-based 유지
                "most_common_wrong_answer": most_common_wrong_answer,
                "most_common_wrong_count": most_common_wrong_count,
                "precision_scores": precision_scores,
                "correct_precision_score": correct_precision_score,
                "max_precision_score": max_precision_score,
                "max_precision_index": max_precision_index,
                "is_max_precision_correct": max_precision_index == correct_indices[0],
            }
            
            single_correct_features.append(feature)
    
    return single_correct_features

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

def analyze_all_files(input_folder):
    """
    주어진 폴더의 모든 JSON 파일에 대해 정답이 하나인 문제 분석을 실행
    """
    all_features = []
    all_model_stats = {}
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"경고: {input_folder}에서 JSON 파일을 찾을 수 없습니다!")
        return all_features, all_model_stats
    
    print(f"총 {len(json_files)}개 파일을 처리합니다.")
    
    for file_path in tqdm(json_files, desc="파일 처리 중"):
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
            else:
                results = data
            
            # 정답이 하나인 문제 분석
            features = analyze_single_correct_problems(results)
            
            # 모델별 통계 계산
            total_is_max_precision_correct = sum(f["is_max_precision_correct"] for f in features)
            avg_most_common_wrong_count = np.mean([f["most_common_wrong_count"] for f in features if f["most_common_wrong_count"] > 0])
            
            all_model_stats[model_name] = {
                "model_name": model_name,
                "total_single_correct_problems": len(features),
                "max_precision_correct_count": total_is_max_precision_correct,
                "max_precision_correct_ratio": total_is_max_precision_correct / len(features) if features else 0,
                "avg_most_common_wrong_count": avg_most_common_wrong_count if not np.isnan(avg_most_common_wrong_count) else 0
            }
            
            # 파일명과 모델명 추가하여 전체 결과에 추가
            for feature in features:
                feature["file_path"] = file_path
                feature["model_name"] = model_name
                all_features.append(feature)
                
            print(f"{model_name}: 정답이 하나인 문제 {len(features)}개 중, "
                  f"최대 Precision 점수를 가진 응답이 정답인 경우: {total_is_max_precision_correct}개 ({all_model_stats[model_name]['max_precision_correct_ratio']:.2%}), "
                  f"오답 중 가장 많이 나온 답변의 평균 개수: {avg_most_common_wrong_count:.2f}")
            
        except Exception as e:
            print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    return all_features, all_model_stats

def calculate_wrong_answer_stats(all_features):
    """
    오답 중 가장 많이 나온 답변의 개수 분포를 분석하는 함수
    """
    # 모델별로 그룹화하여 가장 많이 나온 오답의 개수 분포 계산
    wrong_answer_stats = []
    
    # 모델별로 개수 분포 계산
    model_groups = {}
    for feature in all_features:
        model_name = feature["model_name"]
        count = feature["most_common_wrong_count"]
        
        if model_name not in model_groups:
            model_groups[model_name] = []
            
        model_groups[model_name].append(count)
    
    # 모델별로 개수 레벨(1,2,3,4)에 따른 문제 개수 집계
    for model_name, counts in model_groups.items():
        count_counter = Counter(counts)
        
        for count_value in range(0, 5):  # 0부터 4까지 (가능한 오답 개수)
            stat = {
                "model_name": model_name,
                "wrong_answer_count": count_value,
                "num_problems": count_counter.get(count_value, 0),
                "percentage": count_counter.get(count_value, 0) / len(counts) if counts else 0
            }
            wrong_answer_stats.append(stat)
    
    return wrong_answer_stats

def compare_nli_and_precision(input_folder, output_dir):
    """
    NLI(entailment_min)와 Precision 점수의 성능을 비교하는 함수
    """
    comparison_data = []
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"경고: {input_folder}에서 JSON 파일을 찾을 수 없습니다!")
        return
    
    for file_path in tqdm(json_files, desc="NLI와 Precision 비교 중"):
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
            else:
                results = data
            
            # 단일 문제 분석
            nli_correct = 0
            precision_correct = 0
            both_correct = 0
            only_nli_correct = 0
            only_precision_correct = 0
            
            total_single_correct = 0
            
            for item in results:
                correct_answer = item.get("correct_answer")
                if not correct_answer:
                    continue
                
                # 각 응답에서 정답 추출 및 정답 확인
                correct_counts = 0
                correct_indices = []
                
                for i in range(1, 6):
                    response_key = f"response_{i}"
                    if response_key in item and item[response_key]:
                        answer = extract_numerical_answer(item[response_key])
                        
                        # 정답과 일치하는지 확인
                        if is_answer_correct(answer, correct_answer):
                            correct_counts += 1
                            correct_indices.append(i)
                
                # 정답이 하나인 문제만 분석
                if correct_counts == 1:
                    total_single_correct += 1
                    correct_index = correct_indices[0]
                    
                    # NLI와 Precision 점수 분석
                    max_nli_score = 0
                    max_nli_index = 0
                    max_precision_score = 0
                    max_precision_index = 0
                    
                    for i in range(1, 6):
                        if "similarity_scores" in item and f"question_{i}" in item["similarity_scores"]:
                            # NLI 점수 (entailment_min)
                            nli_score = item["similarity_scores"][f"question_{i}"].get("entailment_min", 0)
                            if nli_score > max_nli_score:
                                max_nli_score = nli_score
                                max_nli_index = i
                            
                            # Precision 점수
                            precision_score = item["similarity_scores"][f"question_{i}"].get("precision", 0)
                            if precision_score > max_precision_score:
                                max_precision_score = precision_score
                                max_precision_index = i
                    
                    # 정답 여부 확인
                    is_nli_correct = (max_nli_index == correct_index)
                    is_precision_correct = (max_precision_index == correct_index)
                    
                    if is_nli_correct:
                        nli_correct += 1
                    
                    if is_precision_correct:
                        precision_correct += 1
                    
                    if is_nli_correct and is_precision_correct:
                        both_correct += 1
                    elif is_nli_correct:
                        only_nli_correct += 1
                    elif is_precision_correct:
                        only_precision_correct += 1
            
            # 비교 데이터 저장
            if total_single_correct > 0:
                comparison_data.append({
                    "model_name": model_name,
                    "total_single_correct": total_single_correct,
                    "nli_correct": nli_correct,
                    "nli_accuracy": nli_correct / total_single_correct,
                    "precision_correct": precision_correct,
                    "precision_accuracy": precision_correct / total_single_correct,
                    "both_correct": both_correct,
                    "both_accuracy": both_correct / total_single_correct,
                    "only_nli_correct": only_nli_correct,
                    "only_nli_accuracy": only_nli_correct / total_single_correct,
                    "only_precision_correct": only_precision_correct,
                    "only_precision_accuracy": only_precision_correct / total_single_correct
                })
        
        except Exception as e:
            print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 비교 결과를 CSV로 저장
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = os.path.join(output_dir, "nli_vs_precision_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"NLI와 Precision 비교 결과가 {comparison_csv_path}에 저장되었습니다.")
        
        # 전체 평균 통계 출력
        avg_nli_accuracy = comparison_df["nli_accuracy"].mean()
        avg_precision_accuracy = comparison_df["precision_accuracy"].mean()
        print(f"\nNLI vs Precision 평균 정확도 비교:")
        print(f"NLI 평균 정확도: {avg_nli_accuracy:.2%}")
        print(f"Precision 평균 정확도: {avg_precision_accuracy:.2%}")
        
        better_metric = "NLI" if avg_nli_accuracy > avg_precision_accuracy else "Precision"
        diff = abs(avg_nli_accuracy - avg_precision_accuracy)
        print(f"{better_metric}가 {diff:.2%}만큼 더 우수한 성능을 보입니다.")
    
    return comparison_data

def main():
    # 입력 폴더 경로
    base_path = "/data3/jykim/Projects/CCQA_official/Benchmarks/MAWPS/Result/"
    INPUT_FOLDER = path.join(base_path, "ccqa_t5_result/precision_nli_separate")
    
    # 출력 폴더 경로
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Analyze/single_correct_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Precision 기반 분석 실행
    all_features, all_model_stats = analyze_all_files(INPUT_FOLDER)
    
    # 모델별 통계를 CSV로 저장
    stats_df = pd.DataFrame(list(all_model_stats.values()))
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'single_correct_precision_model_stats.csv'), index=False)
    
    # 오답 개수 통계 계산 및 저장
    wrong_answer_stats = calculate_wrong_answer_stats(all_features)
    wrong_stats_df = pd.DataFrame(wrong_answer_stats)
    wrong_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'most_common_wrong_answer_stats.csv'), index=False)
    
    # 2. NLI와 Precision 비교 분석
    compare_nli_and_precision(INPUT_FOLDER, OUTPUT_DIR)
    
    # 전체 모델에 대한 오답 개수 분포 평균 출력
    avg_by_counts = wrong_stats_df.groupby('wrong_answer_count')['percentage'].mean()
    print("\n오답 중 가장 많이 나온 답변의 개수별 평균 비율:")
    for count, avg_pct in avg_by_counts.items():
        print(f"가장 많이 나온 오답이 {count}번 출현: {avg_pct:.2%}")
    
    print(f"\n분석 결과가 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"- single_correct_precision_model_stats.csv: Precision 기반 모델별 통계")
    print(f"- most_common_wrong_answer_stats.csv: 오답 개수 통계")
    print(f"- nli_vs_precision_comparison.csv: NLI와 Precision 비교 결과")

if __name__ == "__main__":
    main()