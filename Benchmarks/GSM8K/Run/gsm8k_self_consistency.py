import os
import json
import sys
import csv
import re
import random
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append('/data3/jykim/Projects/CCQA_official')

# self_consistency.py 파일 임포트
from Method.self_consistency import apply_self_consistency

def extract_model_name(filename: str) -> str:
    """
    파일명에서 모델 이름을 추출합니다.
    
    Args:
        filename: 파일명
        
    Returns:
        모델 이름
    """
    # GSM8K_Llama-3.2-3B-Instruct_few_shot_result_self_consistency.json 형식에서 모델명 추출
    match = re.search(r'GSM8K_([^_]+)_(?:few_shot_)?result', filename)
    if match:
        return match.group(1)
    return None

def extract_numerical_answer(response: str) -> Optional[str]:
    """
    답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자(또는 A~E)를 추출하는 함수
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
            # 캡처된 그룹(숫자 혹은 A~E) 추출
            answer = match.group(1).strip()
            # 숫자에 천 단위 구분 콤마(,)가 있으면 제거
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    
    return None

def process_gsm8k_file(file_path: str, output_dir: str) -> Dict:
    """
    GSM8K 파일을 처리하여 self-consistency를 적용하고 정확도를 계산합니다.
    
    Args:
        file_path: 입력 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        처리 결과 정보
    """
    try:
        # 파일명에서 모델명 추출
        filename = os.path.basename(file_path)
        model_name = extract_model_name(filename)
        
        if not model_name:
            print(f"경고: 파일명 '{filename}'에서 모델 이름을 추출할 수 없습니다. 파일명을 그대로 사용합니다.")
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
        
        # 필요한 항목만 추출하여 리스트 생성 (Self-consistency에 필요한 필드만 포함)
        processed_results = []
        
        # CoT 정확도를 위한 변수
        cot_correct_count = 0
        cot_valid_count = 0
        total_items = len(results)
        
        for item in results:
            # 기본 항목들 추가
            processed_item = {
                "question": item.get("question", ""),
            }
            
            # correct_answer 필드 확인 및 추가
            if "correct_answer" in item:
                processed_item["correct_answer"] = item["correct_answer"]
            elif "original_answer" in item:
                processed_item["correct_answer"] = item["original_answer"]
            
            # 모든 응답 필드 확인 및 추가
            for key, value in item.items():
                if key.startswith("response_") and key != "response_time":
                    processed_item[key] = value
                elif key == "all_responses" and isinstance(value, list):
                    # all_responses 배열이 있는 경우 response_1, response_2 등으로 변환
                    for i, response in enumerate(value, 1):
                        processed_item[f"response_{i}"] = response
            
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
        
        # self-consistency 적용
        print(f"모델 {model_name}에 self-consistency 적용 중...")
        consistent_results = apply_self_consistency(processed_results)
        
        # 정확도 계산
        sc_correct_count = 0
        sc_valid_count = 0
        
        for item in consistent_results:
            # self_consistency_answer와 correct_answer 비교
            sc_answer = item.get("self_consistency_answer")
            correct_answer = item.get("correct_answer")
            
            # 유효한 답변이 있는 경우만 계산
            if sc_answer is not None and correct_answer is not None:
                sc_valid_count += 1
                
                # 숫자만 추출하여 비교 (동일한 숫자라면 정답으로 간주)
                sc_numeric = re.sub(r'[^\d.]', '', str(sc_answer).strip())
                correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
                
                try:
                    if sc_numeric and correct_numeric:
                        sc_float = float(sc_numeric)
                        correct_float = float(correct_numeric)
                        if abs(sc_float - correct_float) < 1e-5:
                            sc_correct_count += 1
                    else:
                        # 숫자가 아닌 경우 문자열 비교
                        if str(sc_answer).strip() == str(correct_answer).strip():
                            sc_correct_count += 1
                except ValueError:
                    # 숫자로 변환할 수 없는 경우 단순 문자열 비교
                    if str(sc_answer).strip() == str(correct_answer).strip():
                        sc_correct_count += 1
        
        # Self-consistency 정확도 계산
        sc_accuracy = sc_correct_count / total_items if total_items > 0 else 0
        
        # 시간 정보 업데이트
        time_info["cot_accuracy"] = cot_accuracy
        time_info["cot_correct_count"] = cot_correct_count
        time_info["cot_valid_count"] = cot_valid_count
        time_info["self_consistency_accuracy"] = sc_accuracy
        time_info["self_consistency_valid_count"] = sc_valid_count
        time_info["self_consistency_correct_count"] = sc_correct_count
        
        # 결과 저장
        output_file = os.path.join(output_dir, f"gsm8k_{model_name}_self_consistency.json")
        
        # 원래 형식대로 저장 (time_info가 있었다면 유지)
        if "time_info" in data or time_info:
            output_data = {
                "time_info": time_info,
                "results": consistent_results
            }
        else:
            output_data = consistent_results
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"모델 {model_name} 처리 완료:")
        print(f"  CoT 정확도: {cot_accuracy:.2%} ({cot_correct_count}/{total_items})")
        print(f"  Self-consistency 정확도: {sc_accuracy:.2%} ({sc_correct_count}/{total_items})")
        
        return {
            "path": output_file,
            "model_name": model_name,
            "cot_accuracy": cot_accuracy,
            "cot_correct_count": cot_correct_count,
            "cot_valid_count": cot_valid_count,
            "sc_accuracy": sc_accuracy,
            "sc_correct_count": sc_correct_count,
            "sc_valid_count": sc_valid_count,
            "total_items": total_items
        }
    
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_gsm8k_self_consistency(
    results_dir: str,
    output_dir: str
) -> Dict[str, Dict]:
    """
    GSM8K 벤치마크에 대한 self-consistency 실행
    
    Args:
        results_dir: 원본 GSM8K 결과 파일이 있는 디렉토리
        output_dir: self-consistency 결과를 저장할 디렉토리
        
    Returns:
        모델별 처리 결과 정보
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 GSM8K 파일 목록 가져오기
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'GSM8K' in f]
    
    if not json_files:
        print(f"경고: {results_dir}에서 GSM8K 데이터 파일을 찾을 수 없습니다!")
        return {}
    
    print(f"총 {len(json_files)}개 파일을 처리합니다.")
    
    # 모델별 결과 저장
    model_results = {}
    
    # 각 파일 처리
    for filename in tqdm(json_files, desc="파일 처리 중"):
        file_path = os.path.join(results_dir, filename)
        result = process_gsm8k_file(file_path, output_dir)
        
        if result:
            model_name = result["model_name"]
            model_results[model_name] = result
    
    # 결과 요약 보고
    print("\nSelf-consistency 완료. 결과 요약:")
    
    # CSV로 정확도 정보 요약 저장
    csv_path = os.path.join(output_dir, "self_consistency_accuracy_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # CSV 헤더 정의
        fieldnames = ['model_name', 'cot_accuracy', 'cot_correct_count', 'sc_accuracy', 'sc_correct_count', 'total_items']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 각 모델의 정확도 정보 작성
        for model_name, result in model_results.items():
            if "cot_accuracy" in result and "sc_accuracy" in result:
                writer.writerow({
                    'model_name': model_name,
                    'cot_accuracy': f"{result['cot_accuracy']:.4f}",
                    'cot_correct_count': result['cot_correct_count'],
                    'sc_accuracy': f"{result['sc_accuracy']:.4f}",
                    'sc_correct_count': result['sc_correct_count'],
                    'total_items': result['total_items']
                })
                
                print(f"{model_name}:")
                print(f"  CoT: {result['cot_accuracy']:.2%} ({result['cot_correct_count']}/{result['total_items']})")
                print(f"  SC:  {result['sc_accuracy']:.2%} ({result['sc_correct_count']}/{result['total_items']})")
            else:
                writer.writerow({
                    'model_name': model_name,
                    'cot_accuracy': 'N/A',
                    'cot_correct_count': 'N/A',
                    'sc_accuracy': 'N/A',
                    'sc_correct_count': 'N/A',
                    'total_items': 'N/A'
                })
                
                print(f"{model_name}: 실패")
    
    # JSON으로 정확도 정보 요약 저장
    accuracy_summary = {model: {k: v for k, v in result.items() if k != "path"} 
                        for model, result in model_results.items()}
    
    accuracy_summary_path = os.path.join(output_dir, "self_consistency_accuracy_summary.json")
    with open(accuracy_summary_path, 'w', encoding='utf-8') as f:
        json.dump(accuracy_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")
    print(f"정확도 요약 정보는 JSON 형식으로 {accuracy_summary_path}에 저장되었습니다.")
    print(f"정확도 요약 정보는 CSV 형식으로 {csv_path}에 저장되었습니다.")
    
    return model_results

# 설정 파라미터 - 제공된 경로 사용
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/gsm8k_result"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/self_consistency_result"

# 메인 실행
if __name__ == "__main__":
    run_gsm8k_self_consistency(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR
    )