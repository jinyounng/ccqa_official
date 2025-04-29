import os
import json
import re
import numpy as np
import glob
from collections import Counter
from tqdm import tqdm
import pandas as pd

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

def analyze_nli_first_approach(input_folder, output_dir):
    """
    NLI 점수를 우선적으로 사용하고, NLI 점수가 모두 0인 경우에만 precision 점수를 사용하는 분석
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
            
            total_problems = 0
            precision_correct = 0
            nli_correct = 0
            nli_first_correct = 0
            
            # 각 방식이 선택된 횟수 카운트
            nli_choice_count = 0
            precision_backup_count = 0
            zero_nli_count = 0
            nonzero_nli_count = 0
            
            for item in tqdm(results, desc=f"{model_name} 분석 중", leave=False):
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
                if correct_counts <= 1:
                    total_problems += 1
                    correct_index = correct_indices[0]
                    
                    # 각 응답의 점수 수집
                    precision_scores = {}
                    nli_scores = {}
                    
                    for i in range(1, 6):
                        if "similarity_scores" in item and f"question_{i}" in item["similarity_scores"]:
                            # Precision 점수
                            precision_scores[i] = item["similarity_scores"][f"question_{i}"].get("precision", 0)
                            # NLI 점수
                            nli_scores[i] = item["similarity_scores"][f"question_{i}"].get("entailment_min", 0)
                    
                    if not precision_scores or not nli_scores:
                        continue
                    
                    # precision 최고점
                    precision_sorted = sorted(precision_scores.items(), key=lambda x: x[1], reverse=True)
                    precision_best = precision_sorted[0][0]
                    if precision_best == correct_index:
                        precision_correct += 1
                    
                    # NLI 최고점
                    nli_sorted = sorted(nli_scores.items(), key=lambda x: x[1], reverse=True)
                    nli_best = nli_sorted[0][0]
                    if nli_best == correct_index:
                        nli_correct += 1
                    
                    # NLI 점수가 모두 0인지 확인
                    all_nli_zero = all(score == 0 for score in nli_scores.values())
                    
                    # NLI-first 접근법
                    if all_nli_zero:
                        # NLI 점수가 모두 0이면 precision 점수로 결정
                        nli_first_choice = precision_best
                        precision_backup_count += 1
                        zero_nli_count += 1
                    else:
                        # NLI 점수가 0이 아닌 경우 NLI 최고점으로 결정
                        nli_first_choice = nli_best
                        nli_choice_count += 1
                        nonzero_nli_count += 1
                    
                    # 정확도 확인
                    if nli_first_choice == correct_index:
                        nli_first_correct += 1
            
            # 정확도 계산
            precision_accuracy = precision_correct / total_problems if total_problems > 0 else 0
            nli_accuracy = nli_correct / total_problems if total_problems > 0 else 0
            nli_first_accuracy = nli_first_correct / total_problems if total_problems > 0 else 0
            
            # 각 방식 비율 계산
            nli_choice_ratio = nli_choice_count / total_problems if total_problems > 0 else 0
            precision_backup_ratio = precision_backup_count / total_problems if total_problems > 0 else 0
            zero_nli_ratio = zero_nli_count / total_problems if total_problems > 0 else 0
            nonzero_nli_ratio = nonzero_nli_count / total_problems if total_problems > 0 else 0
            
            # 모델별 결과 저장
            all_model_results[model_name] = {
                "model_name": model_name,
                "total_problems": total_problems,
                "precision_accuracy": precision_accuracy,
                "nli_accuracy": nli_accuracy,
                "nli_first_accuracy": nli_first_accuracy,
                "nli_choice_count": nli_choice_count,
                "nli_choice_ratio": nli_choice_ratio,
                "precision_backup_count": precision_backup_count,
                "precision_backup_ratio": precision_backup_ratio,
                "zero_nli_count": zero_nli_count,
                "zero_nli_ratio": zero_nli_ratio,
                "nonzero_nli_count": nonzero_nli_count,
                "nonzero_nli_ratio": nonzero_nli_ratio
            }
            
            # 결과 출력
            print(f"\n{model_name} 모델 NLI-first 분석 결과:")
            print(f"  총 분석 문제 수: {total_problems}개")
            print(f"  Precision 정확도: {precision_accuracy:.2%} ({precision_correct}/{total_problems})")
            print(f"  NLI 정확도: {nli_accuracy:.2%} ({nli_correct}/{total_problems})")
            print(f"  NLI-first 정확도: {nli_first_accuracy:.2%} ({nli_first_correct}/{total_problems})")
            print(f"  NLI 점수가 모두 0인 문제: {zero_nli_count}개 ({zero_nli_ratio:.2%})")
            print(f"  NLI 점수가 0이 아닌 문제: {nonzero_nli_count}개 ({nonzero_nli_ratio:.2%})")
            print(f"  NLI 점수로 결정한 경우: {nli_choice_count}개 ({nli_choice_ratio:.2%})")
            print(f"  Precision 백업으로 결정한 경우: {precision_backup_count}개 ({precision_backup_ratio:.2%})")
            
        except Exception as e:
            print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과를 CSV로 저장
    if all_model_results:
        comparison_data = []
        for model_name, results in all_model_results.items():
            comparison_data.append(results)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = os.path.join(output_dir, "nli_first_approach.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"\nNLI-first 접근법 비교 결과가 {comparison_csv_path}에 저장되었습니다.")
        
        # 전체 평균 계산
        avg_precision_accuracy = comparison_df["precision_accuracy"].mean()
        avg_nli_accuracy = comparison_df["nli_accuracy"].mean()
        avg_nli_first_accuracy = comparison_df["nli_first_accuracy"].mean()
        avg_zero_nli_ratio = comparison_df["zero_nli_ratio"].mean()
        
        print("\n모든 모델의 평균 정확도:")
        print(f"Precision만: {avg_precision_accuracy:.4f}")
        print(f"NLI만: {avg_nli_accuracy:.4f}")
        print(f"NLI-first 접근법: {avg_nli_first_accuracy:.4f}")
        print(f"NLI 점수가 모두 0인 비율: {avg_zero_nli_ratio:.2%}")
        
        # 성능 향상 계산
        best_single = max(avg_precision_accuracy, avg_nli_accuracy)
        best_single_name = "Precision" if avg_precision_accuracy > avg_nli_accuracy else "NLI"
        
        improvement = avg_nli_first_accuracy - best_single
        if improvement > 0:
            print(f"NLI-first 접근법이 {best_single_name}보다 {improvement:.4f} 향상")
        else:
            print(f"NLI-first 접근법이 단일 점수보다 효과적이지 않음 ({-improvement:.4f} 하락)")
    
    return all_model_results

def main():
    # 입력 폴더 경로
    INPUT_FOLDER = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_0_result/precision_nli_separate"
    
    # 출력 폴더 경로
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Analyze/single_correct_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # NLI 우선, precision 백업 접근법 실행
    analyze_nli_first_approach(INPUT_FOLDER, OUTPUT_DIR)
    
    print(f"\n분석 결과가 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"- nli_first_approach.csv: NLI 우선, Precision 백업 접근법 결과")

if __name__ == "__main__":
    main()