import os
import json
import sys
import csv
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append('/data3/jykim/Projects/CCQA_official')

# 직접 self_consistency.py 파일의 경로를 지정하여 임포트
from Method.self_consistency import *

def run_svamp_self_consistency(
    results_dir: str,
    output_dir: str,
    parallel: bool = False
) -> Dict[str, Dict]:
    benchmark_name = "svamp"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"SVAMP 벤치마크에 대한 self-consistency 시작...")
    results = run_all_models_self_consistency(
        results_dir=results_dir,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
        parallel=parallel
    )
    
    # 결과 요약 보고
    print("\nSelf-consistency 완료. 결과 요약:")
    
    # 모든 모델 정확도 정보를 저장할 딕셔너리
    accuracy_summary = {}
    
    for model_name, result_info in results.items():
        if result_info:
            path = result_info["path"]
            
            # 결과 파일을 로드하여 정확도 계산
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                
                # 결과 데이터 구조 확인
                if "results" in model_results:
                    items = model_results["results"]
                else:
                    items = model_results
                
                total_items = len(items)
                sc_correct_count = 0    # self-consistency 정답 개수
                cot_correct_count = 0   # CoT 정답 개수
                valid_answers = 0
                
                for item in items:
                    # self_consistency_answer와 정답 비교
                    sc_answer = item.get("self_consistency_answer")
                    original_answer = item.get("original_answer")
                    
                    # CoT 정확도를 위한 self_consistency_extraction의 첫 번째 항목 사용
                    sc_extraction = item.get("self_consistency_extraction", [])
                    cot_answer = sc_extraction[0] if sc_extraction and len(sc_extraction) > 0 else None
                    
                    # 유효한 답변이 있는 경우만 계산
                    if original_answer is not None:
                        valid_answers += 1
                        
                        # Self-consistency 정확도 계산
                        if sc_answer is not None:
                            try:
                                # 숫자로 변환할 수 있는 경우 수치적 비교
                                sc_numeric = float(str(sc_answer).strip())
                                original_numeric = float(str(original_answer).strip())
                                
                                # 작은 오차 허용 (부동소수점 비교를 위해)
                                if abs(sc_numeric - original_numeric) < 1e-5:
                                    sc_correct_count += 1
                            except (ValueError, TypeError):
                                # 숫자로 변환할 수 없는 경우 문자열 비교
                                if str(sc_answer).strip() == str(original_answer).strip():
                                    sc_correct_count += 1
                        
                        # CoT 정확도 계산 (self_consistency_extraction의 첫 번째 항목)
                        if cot_answer is not None:
                            try:
                                # 숫자로 변환할 수 있는 경우 수치적 비교
                                cot_numeric = float(str(cot_answer).strip())
                                original_numeric = float(str(original_answer).strip())
                                
                                # 작은 오차 허용 (부동소수점 비교를 위해)
                                if abs(cot_numeric - original_numeric) < 1e-5:
                                    cot_correct_count += 1
                            except (ValueError, TypeError):
                                # 숫자로 변환할 수 없는 경우 문자열 비교
                                if str(cot_answer).strip() == str(original_answer).strip():
                                    cot_correct_count += 1
                
                # 정확도 계산
                sc_accuracy = sc_correct_count / total_items if total_items > 0 else 0
                cot_accuracy = cot_correct_count / total_items if total_items > 0 else 0
                
                # 정확도 정보 저장
                accuracy_summary[model_name] = {
                    "total_items": total_items,
                    "valid_answers": valid_answers,
                    "sc_correct_count": sc_correct_count,
                    "cot_correct_count": cot_correct_count,
                    "sc_accuracy": sc_accuracy,
                    "cot_accuracy": cot_accuracy
                }
                
                # 결과 출력
                print(f"{model_name}:")
                print(f"   - CoT 정확도: {cot_accuracy:.2%} ({cot_correct_count}/{total_items})")
                print(f"   - Self-consistency 정확도: {sc_accuracy:.2%} ({sc_correct_count}/{total_items})")
                print(f"   - 출력 경로: {path}")
                
            except Exception as e:
                print(f"{model_name} 정확도 계산 중 오류 발생: {e}")
                accuracy_summary[model_name] = {
                    "error": str(e)
                }
        else:
            print(f" {model_name}: 실패")
    
    # 모델 성능 비교 테이블 생성 (요청한 형식대로)
    comparison_csv_path = os.path.join(output_dir, "model_comparison.csv")
    with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
        # CSV 헤더 정의 - 요청한 형식
        fieldnames = ['Models', 'CoT-accuracy', 'Self-consistency-accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 각 모델의 정확도 정보 작성
        for model_name, acc_info in accuracy_summary.items():
            if "cot_accuracy" in acc_info and "sc_accuracy" in acc_info:
                writer.writerow({
                    'Models': model_name,
                    'CoT-accuracy': f"{acc_info['cot_accuracy']:.4f}",
                    'Self-consistency-accuracy': f"{acc_info['sc_accuracy']:.4f}"
                })
            else:
                writer.writerow({
                    'Models': model_name,
                    'CoT-accuracy': 'N/A',
                    'Self-consistency-accuracy': 'N/A'
                })
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")
    print(f"모델 비교 정보는 CSV 형식으로 {comparison_csv_path}에 저장되었습니다.")
    
    # 전체 정확도 요약 출력
    print("\n모델별 정확도 요약:")
    for model_name, acc_info in accuracy_summary.items():
        if "cot_accuracy" in acc_info and "sc_accuracy" in acc_info:
            print(f"{model_name}:")
            print(f"  - CoT 정확도: {acc_info['cot_accuracy']:.2%} ({acc_info['cot_correct_count']}/{acc_info['total_items']})")
            print(f"  - Self-consistency 정확도: {acc_info['sc_accuracy']:.2%} ({acc_info['sc_correct_count']}/{acc_info['total_items']})")
    
    return results

# 설정 파라미터
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/svamp_results"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/self_consistency_result"
RUN_PARALLEL = True  # 모델을 병렬로 실행하려면 True로 설정

# 메인 실행
if __name__ == "__main__":
    run_svamp_self_consistency(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        parallel=RUN_PARALLEL
    )