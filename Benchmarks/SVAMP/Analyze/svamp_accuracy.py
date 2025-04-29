import os
import json
import re
import csv
import glob
from typing import Dict, List, Optional, Tuple, Any

def is_answer_correct(extracted_answer: Optional[str], original_answer: str) -> bool:
    if extracted_answer is None:
        return False
        
    # 먼저 직접 문자열 비교
    if extracted_answer == original_answer:
        return True
        
    # 숫자 비교 시도
    try:
        extracted_float = float(extracted_answer)
        original_float = float(original_answer)
        return abs(extracted_float - original_float) < 1e-6
    except (ValueError, TypeError):
        # 변환 실패 시 문자열 비교로 fallback
        return extracted_answer == original_answer

def load_json_file(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"파일 '{file_path}' 로드 중 오류: {e}")
        return {}

def extract_model_name(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    
    # svamp_모델명_* 패턴 매칭
    match = re.search(r'svamp_(.+?)_', file_name)
    if match:
        return match.group(1)
    else:
        return "unknown"

def load_self_consistency_data(consistency_dir: str, model_name: str) -> List:
    """
    자가 일관성 데이터 로드
    
    Args:
        consistency_dir: 자가 일관성 결과 디렉토리
        model_name: 모델 이름
        
    Returns:
        자가 일관성 결과 데이터
    """
    consistency_file = os.path.join(consistency_dir, f"svamp_{model_name}_self_consistency.json")
    
    if os.path.exists(consistency_file):
        data = load_json_file(consistency_file)
        if "results" in data:
            return data["results"]
        return data
    else:
        print(f"자가 일관성 파일을 찾을 수 없음: {consistency_file}")
        return []

def calculate_accuracies():
    # 디렉토리 경로 설정
    base_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results"
    self_correction_dir = os.path.join(base_dir, "self_correction_result")
    ccqa_dir = os.path.join(base_dir, "ccqa_t5_result")
    self_consistency_dir = os.path.join(base_dir, "self_consistency_result")
    output_dir = os.path.join(base_dir, "accuracy_results")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 저장을 위한 데이터 구조
    all_results = []
    detailed_results = {}
    
    # 자가 수정 파일 처리
    correction_files = glob.glob(os.path.join(self_correction_dir, "*_refined.json"))
    
    for correction_file in correction_files:
        model_name = extract_model_name(correction_file)
        print(f"\n모델 '{model_name}' 처리 중...")
        
        # 자가 수정 데이터 로드
        correction_data = load_json_file(correction_file)
        if "results" in correction_data:
            correction_results = correction_data["results"]
        else:
            correction_results = correction_data
            
        # CCQA 파일 찾기
        ccqa_file = os.path.join(ccqa_dir, f"svamp_{model_name}_ccqa_t5.json")
        if not os.path.exists(ccqa_file):
            print(f"CCQA 파일을 찾을 수 없음: {ccqa_file}")
            ccqa_results = []
        else:
            ccqa_data = load_json_file(ccqa_file)
            if "results" in ccqa_data:
                ccqa_results = ccqa_data["results"]
            else:
                ccqa_results = ccqa_data
        
        # 자가 일관성 데이터 로드
        consistency_results = load_self_consistency_data(self_consistency_dir, model_name)
        
        # 정확도 카운터 초기화
        total_items = len(correction_results)
        cot_correct = 0
        self_correction_correct = 0
        self_consistency_correct = 0
        ccqa_bertscore_correct = 0
        ccqa_llm_correct = 0
        
        # 상세 결과 저장
        model_details = []
        
        # 각 문항 처리
        for i, item in enumerate(correction_results):
            if i >= len(ccqa_results) or i >= len(consistency_results):
                continue
                
            ccqa_item = ccqa_results[i] if i < len(ccqa_results) else {}
            consistency_item = consistency_results[i] if i < len(consistency_results) else {}
            
            # 원본 답변
            original_answer = item.get("original_answer", "")
            
            # 각 방법의 답변 확인
            
            # 1. CoT (첫번째 응답)
            response_1 = item.get("response_1", "")
            cot_answer = extract_answer_from_text(response_1)
            is_cot_correct = is_answer_correct(cot_answer, original_answer)
            if is_cot_correct:
                cot_correct += 1
            
            # 2. Self-correction
            self_correction_answer = item.get("self_correction_answer", None)
            is_self_correction_correct = is_answer_correct(self_correction_answer, original_answer)
            if is_self_correction_correct:
                self_correction_correct += 1
            
            # 3. Self-consistency
            self_consistency_answer = consistency_item.get("self_consistency_answer", None)
            is_self_consistency_correct = is_answer_correct(self_consistency_answer, original_answer)
            if is_self_consistency_correct:
                self_consistency_correct += 1
            
            # 4. CCQA BERTScore
            ccqa_bertscore_answer = ccqa_item.get("ccqa_bertscore_answer", None)
            is_ccqa_bertscore_correct = is_answer_correct(ccqa_bertscore_answer, original_answer)
            if is_ccqa_bertscore_correct:
                ccqa_bertscore_correct += 1
            
            # 5. CCQA LLM
            ccqa_llm_answer = ccqa_item.get("extracted_answer", None)
            is_ccqa_llm_correct = is_answer_correct(ccqa_llm_answer, original_answer)
            if is_ccqa_llm_correct:
                ccqa_llm_correct += 1
            
            # 상세 정보 저장
            detail = {
                "question": item.get("body", "") + " " + item.get("question", ""),
                "original_answer": original_answer,
                "cot_answer": cot_answer,
                "self_correction_answer": self_correction_answer,
                "self_consistency_answer": self_consistency_answer,
                "is_self_correction_correct": is_self_correction_correct,
                "is_self_consistency_correct": is_self_consistency_correct,
                "is_ccqa_bertscore_correct": is_ccqa_bertscore_correct,
                "is_ccqa_llm_correct": is_ccqa_llm_correct
            }
            model_details.append(detail)
        
        # 정확도 계산
        cot_accuracy = cot_correct / total_items if total_items > 0 else 0
        self_correction_accuracy = self_correction_correct / total_items if total_items > 0 else 0
        self_consistency_accuracy = self_consistency_correct / total_items if total_items > 0 else 0
        ccqa_bertscore_accuracy = ccqa_bertscore_correct / total_items if total_items > 0 else 0
        ccqa_llm_accuracy = ccqa_llm_correct / total_items if total_items > 0 else 0
        
        # 결과 출력
        print(f"모델: {model_name}")
        print(f"  문항 수: {total_items}")
        print(f"  CoT 정확도: {cot_accuracy:.4f} ({cot_correct}/{total_items})")
        print(f"  Self-correction 정확도: {self_correction_accuracy:.4f} ({self_correction_correct}/{total_items})")
        print(f"  Self-consistency 정확도: {self_consistency_accuracy:.4f} ({self_consistency_correct}/{total_items})")
        print(f"  CCQA BERTScore 정확도: {ccqa_bertscore_accuracy:.4f} ({ccqa_bertscore_correct}/{total_items})")
        print(f"  CCQA LLM 정확도: {ccqa_llm_accuracy:.4f} ({ccqa_llm_correct}/{total_items})")
        
        # 모델 결과 저장
        result = {
            "model": model_name,
            "total_items": total_items,
            "cot_accuracy": cot_accuracy,
            "self_correction_accuracy": self_correction_accuracy,
            "self_consistency_accuracy": self_consistency_accuracy,
            "ccqa_llm_accuracy": ccqa_llm_accuracy
        }
        all_results.append(result)
        
        # 상세 결과 저장
        detailed_results[model_name] = model_details
        
        # 개별 모델 상세 파일 저장
        detail_file = os.path.join(output_dir, f"{model_name}_detailed.json")
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(model_details, f, ensure_ascii=False, indent=2)
    
    # 모든 모델 결과를 CSV로 저장
    csv_file = os.path.join(output_dir, "all_accuracies.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "model", "total_items", 
            "cot_accuracy", "self_correction_accuracy", "self_consistency_accuracy", 
            "ccqa_llm_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    # 평균 정확도 계산
    avg_cot = sum(r["cot_accuracy"] for r in all_results) / len(all_results) if all_results else 0
    avg_self_correction = sum(r["self_correction_accuracy"] for r in all_results) / len(all_results) if all_results else 0
    avg_self_consistency = sum(r["self_consistency_accuracy"] for r in all_results) / len(all_results) if all_results else 0
    avg_ccqa_llm = sum(r["ccqa_llm_accuracy"] for r in all_results) / len(all_results) if all_results else 0
    
    # 요약 정보 저장
    summary = {
        "models": all_results,
        "averages": {
            "cot_accuracy": avg_cot,
            "self_correction_accuracy": avg_self_correction,
            "self_consistency_accuracy": avg_self_consistency,
            "ccqa_llm_accuracy": avg_ccqa_llm
        }
    }
    
    summary_file = os.path.join(output_dir, "accuracy_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 결과 요약 출력
    print("\n평균 정확도:")
    print(f"  CoT: {avg_cot:.4f}")
    print(f"  Self-correction: {avg_self_correction:.4f}")
    print(f"  Self-consistency: {avg_self_consistency:.4f}")
    print(f"  CCQA LLM: {avg_ccqa_llm:.4f}")
    
    print(f"\n결과가 '{output_dir}'에 저장되었습니다.")

def extract_answer_from_text(text: str) -> Optional[str]:
    """
    텍스트에서 숫자 형태의 답변 추출
    
    Args:
        text: 텍스트
        
    Returns:
        추출된 답변 또는 None
    """
    patterns = [
        r'the answer is (\d+\.?\d*)',
        r'answer is (\d+\.?\d*)',
        r'answer: (\d+\.?\d*)',
        r'(\d+\.?\d*) is the answer',
        r'(\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

if __name__ == "__main__":
    calculate_accuracies()