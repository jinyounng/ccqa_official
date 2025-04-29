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

def compute_combined_score(precision_score, nli_score, alpha):
    """
    Precision과 NLI 점수를 가중합하여 결합 점수를 계산
    
    formula: precision_score * (1 - alpha) + nli_score * alpha
    
    Args:
        precision_score: Precision 점수
        nli_score: NLI 점수 (entailment_min)
        alpha: NLI 가중치 (0~1 사이 값)
        
    Returns:
        결합된 점수
    """
    return precision_score * (1 - alpha) + nli_score * alpha

def analyze_with_combined_scores(results, alpha_values):
    """
    다양한 alpha 값에 대해 precision과 NLI 가중합 점수로 분석
    
    Args:
        results: 원본 결과 데이터
        alpha_values: 테스트할 alpha 값들의 리스트
        
    Returns:
        모델별, alpha별 정확도 통계 및 모든 문제의 분석 결과
    """
    alpha_stats = {alpha: {"correct_count": 0, "total_count": 0} for alpha in alpha_values}
    all_problem_results = []
    
    for item in tqdm(results, desc="가중합 점수 분석 중"):
        correct_answer = item.get("original_answer")
        if not correct_answer:
            continue
            
        # 각 응답에서 정답 추출
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
            correct_index = correct_indices[0]
            problem_result = {
                "question_id": item.get("id", "unknown"),
                "correct_answer": correct_answer,
                "correct_index": correct_index,
                "alpha_results": {}
            }
            
            # 각 응답의 NLI와 Precision 점수 수집
            responses_scores = []
            
            for i in range(1, 6):
                if "similarity_scores" in item and f"question_{i}" in item["similarity_scores"]:
                    # Precision 점수
                    precision_score = item["similarity_scores"][f"question_{i}"].get("precision", 0)
                    # NLI 점수 (entailment_min)
                    nli_score = item["similarity_scores"][f"question_{i}"].get("entailment_min", 0)
                    
                    responses_scores.append({
                        "index": i,
                        "precision_score": precision_score,
                        "nli_score": nli_score,
                        "is_correct": (i == correct_index)
                    })
            
            # 각 alpha 값에 대해 가중합 점수 계산 및 정확도 측정
            for alpha in alpha_values:
                # 가중합 점수 계산
                for response in responses_scores:
                    response[f"combined_score_{alpha}"] = compute_combined_score(
                        response["precision_score"], 
                        response["nli_score"], 
                        alpha
                    )
                
                # 가장 높은 가중합 점수를 가진 응답 찾기
                if responses_scores:
                    # alpha 값별로 정렬
                    sorted_responses = sorted(
                        responses_scores, 
                        key=lambda x: x[f"combined_score_{alpha}"], 
                        reverse=True
                    )
                    best_response = sorted_responses[0]
                    
                    # 정확도 계산
                    is_correct = best_response["is_correct"]
                    alpha_stats[alpha]["total_count"] += 1
                    if is_correct:
                        alpha_stats[alpha]["correct_count"] += 1
                    
                    # 문제별 결과 저장
                    problem_result["alpha_results"][str(alpha)] = {
                        "best_index": best_response["index"],
                        "best_score": best_response[f"combined_score_{alpha}"],
                        "is_correct": is_correct
                    }
            
            all_problem_results.append(problem_result)
    
    # 각 alpha에 대한 정확도 계산
    alpha_accuracies = {}
    for alpha, stats in alpha_stats.items():
        if stats["total_count"] > 0:
            alpha_accuracies[alpha] = stats["correct_count"] / stats["total_count"]
        else:
            alpha_accuracies[alpha] = 0
    
    return alpha_accuracies, all_problem_results

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

def analyze_all_files_with_combined_scores(input_folder, alpha_values):
    """
    주어진 폴더의 모든 JSON 파일에 대해 가중합 점수 분석을 실행
    """
    all_model_results = {}
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"경고: {input_folder}에서 JSON 파일을 찾을 수 없습니다!")
        return all_model_results
    
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
            
            # 가중합 점수 분석 실행
            alpha_accuracies, problem_results = analyze_with_combined_scores(results, alpha_values)
            
            # 모델별 결과 저장
            all_model_results[model_name] = {
                "model_name": model_name,
                "alpha_accuracies": alpha_accuracies,
                "problem_count": len(problem_results),
                "problem_results": problem_results
            }
            
            # 결과 출력
            print(f"\n{model_name} 모델 분석 결과:")
            print(f"  총 분석 문제 수: {len(problem_results)}개")
            for alpha, accuracy in alpha_accuracies.items():
                correct_count = int(accuracy * len(problem_results))
                print(f"  alpha={alpha}: 정확도 {accuracy:.2%} ({correct_count}/{len(problem_results)})")
            
        except Exception as e:
            print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    return all_model_results

def compare_scores_methods(input_folder, output_dir):
    """
    Precision, NLI, 가중합 점수의 성능을 비교하는 함수
    """
    # 테스트할 alpha 값들
    alpha_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    # 가중합 점수로 모든 파일 분석
    all_model_results = analyze_all_files_with_combined_scores(input_folder, alpha_values)
    
    # 비교 결과를 저장할 DataFrame 생성
    comparison_data = []
    
    for model_name, model_result in all_model_results.items():
        row = {
            "model_name": model_name,
            "problem_count": model_result["problem_count"]
        }
        
        # 각 alpha 값에 대한 정확도 추가
        for alpha, accuracy in model_result["alpha_accuracies"].items():
            row[f"alpha_{alpha}"] = accuracy
        
        comparison_data.append(row)
    
    # 결과를 CSV로 저장
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = os.path.join(output_dir, "combined_scores_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"\n가중합 점수 비교 결과가 {comparison_csv_path}에 저장되었습니다.")
        
        # 전체 평균 통계 출력
        print("\n모든 모델의 평균 정확도:")
        for alpha in alpha_values:
            avg_accuracy = comparison_df[f"alpha_{alpha}"].mean()
            print(f"alpha={alpha}: {avg_accuracy:.4f}")
        
        # 가장 좋은 alpha 값 찾기
        best_alpha_col = max(
            [f"alpha_{alpha}" for alpha in alpha_values],
            key=lambda col: comparison_df[col].mean()
        )
        best_alpha = best_alpha_col.replace("alpha_", "")
        best_avg_accuracy = comparison_df[best_alpha_col].mean()
        
        print(f"\n가장 좋은 가중치: alpha={best_alpha} (평균 정확도: {best_avg_accuracy:.4f})")
        
        # Precision만 사용할 때와 NLI만 사용할 때 비교
        precision_accuracy = comparison_df["alpha_0.0"].mean()
        nli_accuracy = comparison_df["alpha_1.0"].mean()
        combined_best_accuracy = best_avg_accuracy
        
        print("\n점수 방식별 비교:")
        print(f"Precision만 사용: {precision_accuracy:.4f}")
        print(f"NLI만 사용: {nli_accuracy:.4f}")
        print(f"최적 가중합(alpha={best_alpha}): {combined_best_accuracy:.4f}")
        
        # 성능 향상 계산
        best_single = max(precision_accuracy, nli_accuracy)
        best_single_name = "Precision" if precision_accuracy > nli_accuracy else "NLI"
        
        improvement = combined_best_accuracy - best_single
        if improvement > 0:
            print(f"가중합 사용 시 {best_single_name}보다 {improvement:.4f} 향상")
        else:
            print(f"가중합이 단일 점수보다 효과적이지 않음")
    
    return all_model_results

def main():
    # 입력 폴더 경로
    base_path = "/data3/jykim/Projects/CCQA_official/Benchmarks/MAWPS/Result/"
    
    INPUT_FOLDER = path.join(base_path, "ccqa_result/precision_nli_separate")
    
    # 출력 폴더 경로
    OUTPUT_DIR = path.join(base_path,"ccqa_result/precision_nli_separate_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Precision과 NLI 가중합 점수 비교 분석
    compare_scores_methods(INPUT_FOLDER, OUTPUT_DIR)
    
    print(f"\n분석 결과가 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"- combined_scores_comparison.csv: Precision과 NLI 가중합 비교 결과")

if __name__ == "__main__":
    main()