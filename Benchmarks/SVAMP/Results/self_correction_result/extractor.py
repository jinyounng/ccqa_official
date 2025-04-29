import os
import json
import glob
import csv
import re
from typing import Dict, List, Optional

def extract_model_name_from_filename(filename: str) -> Optional[str]:
    """파일명에서 모델 이름을 추출합니다."""
    match = re.search(r'svamp_([^_]+)_refined', filename)
    if match:
        return match.group(1)
    return None

def calculate_accuracy(file_path: str) -> Dict:
    """파일에서 original_answer와 self_correction_answer를 비교하여 정확도를 계산합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"파일 로드 오류 {file_path}: {e}")
        return None
    
    # 결과가 dictionary 형태로 저장되어 있는지 확인
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data
    
    # 실제 데이터셋의 총 문제 수 계산 (SVAMP는 보통 300문제)
    total_count = len(results)
    correct_count = 0
    valid_answers = 0  # 유효한 self_correction_answer 수
    
    for item in results:
        # self_correction_answer가 있고 None이 아닌 경우만 유효 답변으로 처리
        if "original_answer" in item and "self_correction_answer" in item and item["self_correction_answer"] is not None:
            valid_answers += 1
            
            # 답변이 문자열이 아니면 문자열로 변환 (예: 숫자의 경우)
            orig_answer = str(item["original_answer"]).strip()
            sc_answer = str(item["self_correction_answer"]).strip()
            
            # 두 답변이 일치하는지 확인 (간단한 비교)
            if orig_answer == sc_answer:
                correct_count += 1
    
    # 정확도 계산 (전체 문제 수 기준)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    return {
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy
    }

def main():
    # Self-correction 결과 파일이 있는 디렉터리
    correction_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/self_correction_result"
    
    # 출력 CSV 파일 경로
    output_csv = os.path.join(correction_dir, "self_correction_accuracy.csv")
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(correction_dir, "*_refined.json"))
    
    # 모델별 정확도 저장
    model_accuracy = {}
    
    # 각 파일을 처리
    for file_path in json_files:
        filename = os.path.basename(file_path)
        model_name = extract_model_name_from_filename(filename)
        
        if not model_name:
            continue
        
        # 정확도 계산
        accuracy_data = calculate_accuracy(file_path)
        
        if accuracy_data:
            model_accuracy[model_name] = accuracy_data
            print(f"모델 {model_name}의 정확도: {accuracy_data['accuracy']:.2%} ({accuracy_data['correct_count']}/{accuracy_data['total_count']})")
    
    # CSV 파일에 결과 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'accuracy', 'correct_count', 'total_count', 'valid_answers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model_name, data in model_accuracy.items():
            writer.writerow({
                'model_name': model_name,
                'accuracy': data['accuracy'],
                'correct_count': data['correct_count'],
                'total_count': data['total_count'],
                'valid_answers': data.get('valid_answers', 0)
            })
    
    print(f"결과가 {output_csv}에 저장되었습니다.")

if __name__ == "__main__":
    main()