import os
import sys
import json
import re
import torch
import glob
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

# LLM Runner 임포트 (기존 코드 재사용)
sys.path.append("/data3/jykim/Projects/CCQA_official")
from LLM_runner import LLMRunner

# 상수 정의 - SVAMP 경로로 변경
BASE_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result"
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "similar_check")
T5_MODEL_PATH = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-answer-extractor"

# 실행 모드 설정 (True/False로 기능 활성화/비활성화)
RUN_SIMILARITY = True  # 유사성 평가 실행 여부
RUN_EXTRACTION = False   # 정답 추출 실행 여부

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_model_name(filename: str) -> str:
    """파일명에서 모델명 추출"""
    # SVAMP 파일명 패턴으로 업데이트
    pattern = r'svamp_([^_]+)_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    # 새로운 패턴으로 시도
    pattern = r'SVAMP_([^_]+)_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
        
    return "unknown-model"

def evaluate_similarity(
    llm_runner, 
    original_question: str, 
    generated_questions: List[str]
) -> Tuple[List[int], str]:
    """
    주어진 LLM을 사용하여 원본 질문과 가장 유사한 생성 질문(들) 평가
    One-shot prompting 방식 적용
    """
    # 생성된 옵션 포맷팅
    generated_options = ""
    for i, question in enumerate(generated_questions, 1):
        if not question:
            generated_options += f"option {i}: [No question generated]\n"
        else:
            generated_options += f"option {i}: {question}\n"
    
    # One-shot prompting 적용 (예시 제공)
    prompt = f"""
    Look at the original and the options below. 
    List the TOP 5 options that contain numbers and mathematical operations in order of their similarity to the original question, from most similar to least similar. 
    Provide the list as a comma-separated sequence of option numbers (e.g., [3,1,4,5,2]). PLEASE ANSWER the option LIST FIRST, and then explain your reasoning.
    
    Original: "{original_question}"
    Options:
    {generated_options}
    Answer:
    """
    # 응답 생성
    full_response = llm_runner.generate_responses(
        prompt, 
        num_responses=1,
        max_new_tokens=50,
        temperature=0.3,
        top_p=0.9,
        parallel=False
    )[0]
    
    # 우선 콤마로 구분된 숫자 패턴 찾기 (예: "1,3,4,5,2" 또는 "1, 3, 4" 등)
    comma_pattern = re.search(r'(\d+(?:\s*,\s*\d+)+)', full_response)
    
    if comma_pattern:
        # 콤마로 구분된 숫자 추출
        numbers_str = comma_pattern.group(0)
        # 숫자만 추출하여 리스트로 변환
        selected_options = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
    else:
        # 콤마로 구분된 패턴이 없으면 단순히 응답에서 1-5 사이의 숫자를 순서대로 추출
        all_numbers = re.findall(r'\b([1-5])\b', full_response)
        selected_options = [int(n) for n in all_numbers]
    
    # 중복 제거 (순서 유지)
    seen = set()
    selected_options = [x for x in selected_options if not (x in seen or seen.add(x))]
    
    # 유효한 숫자만 포함 (1-5 범위)
    selected_options = [x for x in selected_options if 1 <= x <= 5]
    
    # 최대 5개로 제한
    selected_options = selected_options[:5]
    
    # 5개가 안 되면 기본값으로 채움
    if len(selected_options) < 5:
        # 빠진 숫자들 (1-5 중에 selected_options에 없는 숫자들)
        missing_numbers = [i for i in range(1, 6) if i not in selected_options]
        # 부족한 만큼 채움
        selected_options.extend(missing_numbers[:5-len(selected_options)])
    
    return selected_options, full_response

def extract_answer_with_t5(
    item: Dict[str, Any],
    t5_model,
    t5_tokenizer
) -> Tuple[Optional[str], Optional[str]]:
    """
    T5 모델을 사용하여 응답에서 정답 추출
    단일 인덱스 또는 여러 인덱스를 처리할 수 있음
    """
    # 처리할 인덱스 결정
    similar_idx = None
    
    # most_similar_idx가 있는 경우 (단일 인덱스)
    if "most_similar_idx" in item and item["most_similar_idx"]:
        similar_idx = item["most_similar_idx"]
    # most_similar_idxs가 있는 경우 (첫 번째 인덱스 사용)
    elif "most_similar_idxs" in item and item["most_similar_idxs"]:
        similar_idx = item["most_similar_idxs"][0]
    
    # 인덱스가 없거나 질문이 없으면 None 반환
    if similar_idx is None or "question" not in item:
        print(f"DEBUG: 유효한 인덱스를 찾을 수 없거나 question 필드가 없음")
        return None, None
    
    question = item["question"]
    
    # 인덱스가 범위를 벗어나면 None 반환
    if similar_idx < 1 or similar_idx > 5:
        print(f"DEBUG: 유효하지 않은 인덱스: {similar_idx}")
        return None, None
    
    # 유사한 응답 찾기
    response_key = f"response_{similar_idx}"
    if response_key not in item or not item[response_key]:
        print(f"DEBUG: {response_key} 필드가 없거나 비어 있음")
        return None, None
    
    response = item[response_key]
    
    try:
        # T5 모델용 프롬프트 생성
        prompt = f"Extract the numerical answer from this solution. Question: {question} Solution: {response}"
        
        # 토크나이징 및 추론 실행
        inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # GPU가 있으면 GPU로 입력 데이터 이동
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs, 
                max_length=64, 
                num_beams=4, 
                early_stopping=True
            )
            
        # 출력 디코딩
        extracted_answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        return extracted_answer, response
        
    except Exception as e:
        print(f"T5 모델 추론 중 오류 발생: {str(e)}")
        return None, None

def extract_answers_from_multiple_options(
    item: Dict[str, Any],
    t5_model,
    t5_tokenizer
) -> List[Dict[str, Any]]:
    """
    여러 유사 옵션에서 정답 추출 - 순위 순서 유지
    """
    results = []
    
    # 처리할 인덱스 목록 가져오기 (순위 순서 유지)
    similar_indices = []
    
    if "most_similar_idxs" in item and item["most_similar_idxs"]:
        similar_indices = item["most_similar_idxs"]
    elif "most_similar_idx" in item and item["most_similar_idx"]:
        similar_indices = [item["most_similar_idx"]]
    
    # 인덱스가 없거나 질문이 없으면 빈 결과 반환
    if not similar_indices or "question" not in item:
        return results
    
    question = item["question"]
    
    # 각 인덱스에 대해 정답 추출 (순위 순서대로)
    for rank, idx in enumerate(similar_indices, 1):
        # 인덱스가 범위를 벗어나면 건너뛰기
        if idx < 1 or idx > 5:
            continue
        
        # 해당 응답 찾기
        response_key = f"response_{idx}"
        if response_key not in item or not item[response_key]:
            continue
        
        response = item[response_key]
        
        try:
            # T5 모델용 프롬프트 생성
            prompt = f"Extract the numerical answer from this solution. Question: {question} Solution: {response}"
            
            # 토크나이징 및 추론 실행
            inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # GPU가 있으면 GPU로 입력 데이터 이동
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                
            with torch.no_grad():
                outputs = t5_model.generate(
                    inputs, 
                    max_length=64, 
                    num_beams=4, 
                    early_stopping=True
                )
                
            # 출력 디코딩
            extracted_answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 결과 저장 (랭크 정보 추가)
            if extracted_answer:
                results.append({
                    "idx": idx,
                    "rank": rank,  # 순위 정보 추가
                    "answer": extracted_answer,
                    "source": response
                })
            
        except Exception as e:
            print(f"인덱스 {idx}에 대한 T5 모델 추론 중 오류 발생: {str(e)}")
    
    return results

def process_json_file(json_file_path: str):
    """단일 JSON 파일 처리 및 유사한 질문 저장"""
    filename = os.path.basename(json_file_path)
    print(f"처리 중: {filename}...")
    
    # JSON 데이터 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 결과 배열 가져오기
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data
    
    if not results:
        print(f"{json_file_path}에서 결과를 찾을 수 없습니다.")
        return

    # 필요한 모델 로드
    llm_runner = None
    t5_model = None
    t5_tokenizer = None
    
    if RUN_SIMILARITY:
        # 모델명 추출
        model_name = extract_model_name(filename)
        print(f"파일에서 추출한 모델명: {model_name}")
        
        try:
            llm_runner = LLMRunner(model_name)
            print(f"{model_name} 모델 로드 완료")
        except ValueError as e:
            print(f"모델 로드 오류: {e}")
            print(f"모델이 지원되지 않아 기본 모델(llama-1b)로 대체합니다.")
            llm_runner = LLMRunner("llama-1b")
    
    if RUN_EXTRACTION:
        # T5 모델 로드
        print(f"T5 모델 로드 중: {T5_MODEL_PATH}")
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
        t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
        
        # GPU 사용 가능한 경우 모델을 GPU로 이동
        if torch.cuda.is_available():
            t5_model = t5_model.cuda()
            print("T5 모델이 GPU로 이동되었습니다.")
        else:
            print("GPU를 사용할 수 없어 CPU에서 실행됩니다.")
        
        print("T5 모델 로드 완료")
    
    # 각 항목 처리
    updated_results = []
    extraction_count = 0  # 추출 성공 카운터 추가
    extraction_from_multiple = 0  # 여러 옵션에서 추출 성공 카운터
    
    for i, item in enumerate(tqdm(results, desc=f"{filename} 처리 중")):
        updated_item = item.copy()
        
        # 원본 질문이 없으면 건너뛰기
        if "question" not in item or not item["question"]:
            updated_results.append(updated_item)
            continue
        
        # 유사성 평가 실행
        if RUN_SIMILARITY:
            # 생성된 질문 수집
            generated_questions = []
            for idx in range(1, 6):
                key = f"generated_question_{idx}"
                if key in item and item[key]:
                    generated_questions.append(item[key])
                else:
                    generated_questions.append("")
            
            # 유효한 질문이 충분하지 않으면 건너뛰기
            valid_questions = [q for q in generated_questions if q]
            if len(valid_questions) < 2:
                print(f"항목 {i}에 유효한 질문이 충분하지 않습니다.")
                updated_results.append(updated_item)
                continue
            
            # 가장 유사한 질문 찾기
            try:
                most_similar_idxs, similarity_response = evaluate_similarity(
                    llm_runner,
                    item["question"],
                    generated_questions
                )
                
                # 결과 저장 - 여러 옵션 및 응답만 저장
                updated_item["most_similar_idxs"] = most_similar_idxs  # 순위 순서대로 저장된 리스트
                updated_item["primary_similar_idx"] = most_similar_idxs[0] if most_similar_idxs else 1  # 첫 번째 옵션을 기본으로
                updated_item["similarity_response"] = similarity_response  # 프롬프트 없는 응답
                
                # 이전 버전과의 호환성 유지
                updated_item["most_similar_idx"] = most_similar_idxs[0] if most_similar_idxs else 1
                
            except Exception as e:
                print(f"항목 {i}의 유사성 평가 중 오류 발생: {str(e)}")
                updated_item["similarity_error"] = str(e)
        
        # 정답 추출 실행
        if RUN_EXTRACTION:
            # 여러 유사 인덱스가 있는 경우
            if "most_similar_idxs" in updated_item and updated_item["most_similar_idxs"]:
                # 모든 유사 인덱스에서 정답 추출 (순위 순서대로)
                all_extracted_answers = extract_answers_from_multiple_options(
                    updated_item,
                    t5_model,
                    t5_tokenizer
                )
                
                # 유효한 추출 결과가 있으면 저장
                if all_extracted_answers:
                    updated_item["all_extracted_answers"] = all_extracted_answers
                    
                    # 기본값은 첫 번째 추출 결과 (랭크 1) (이전 버전과의 호환성)
                    updated_item["extracted_answer"] = all_extracted_answers[0]["answer"]
                    updated_item["extraction_source"] = all_extracted_answers[0]["source"]
                    updated_item["extraction_source_idx"] = all_extracted_answers[0]["idx"]
                    updated_item["extraction_rank"] = 1  # 명시적으로 랭크 1 사용
                    
                    extraction_count += 1
                    
                    # 여러 결과가 있는 경우 카운터 증가
                    if len(all_extracted_answers) > 1:
                        extraction_from_multiple += 1
            
            # 이전 버전 호환성 - most_similar_idx가 있는 경우
            elif "most_similar_idx" in updated_item:
                try:
                    # T5 모델로 정답 추출
                    extracted_answer, used_response = extract_answer_with_t5(
                        updated_item,
                        t5_model,
                        t5_tokenizer
                    )
                    
                    # T5 결과 저장 (기존 항목에 추가)
                    if extracted_answer is not None:
                        updated_item["extracted_answer"] = extracted_answer
                        updated_item["extraction_source"] = used_response
                        updated_item["extraction_rank"] = 1  # 단일 결과는 항상 랭크 1
                        extraction_count += 1
                
                except Exception as e:
                    print(f"항목 {i}의 정답 추출 중 오류 발생: {str(e)}")
                    updated_item["extraction_error"] = str(e)
            else:
                print(f"항목 {i}에 most_similar_idx 또는 most_similar_idxs가 없어 정답 추출을 건너뜁니다.")
        
        updated_results.append(updated_item)
        
        # 정기적으로 진행 상황 저장
        save_interval = 10
        if (i + 1) % save_interval == 0 or i == 0 or i == len(results) - 1:
            # 출력 파일명 생성
            output_filename = f"extracted_{filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # time_info가 있으면 유지
            output_data = {}
            if isinstance(data, dict):
                for key in data:
                    if key != "results":
                        output_data[key] = data[key]
            
            # 결과 추가
            output_data["results"] = updated_results
            
            # 추출 정보 추가
            if "time_info" not in output_data:
                output_data["time_info"] = {}
            output_data["time_info"]["extraction_count"] = extraction_count
            output_data["time_info"]["extraction_from_multiple"] = extraction_from_multiple
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            if (i + 1) % save_interval == 0:
                print(f"진행 상황: {i + 1}/{len(results)} 질문 ({((i + 1) / len(results) * 100):.1f}%)")
                print(f"추출 성공: {extraction_count}개, 다중 옵션 추출: {extraction_from_multiple}개")
    
    # 최종 저장
    output_filename = f"extracted_{filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # time_info가 있으면 유지
    output_data = {}
    if isinstance(data, dict):
        for key in data:
            if key != "results":
                output_data[key] = data[key]
    
    # 결과 추가
    output_data["results"] = updated_results
    
    # 추출 정보 추가
    if "time_info" not in output_data:
        output_data["time_info"] = {}
    output_data["time_info"]["extraction_count"] = extraction_count
    output_data["time_info"]["extraction_from_multiple"] = extraction_from_multiple
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"{filename} 처리 완료. 총 {len(results)}개 항목 중 {extraction_count}개 정답 추출 성공")
    if extraction_from_multiple > 0:
        print(f"이 중 {extraction_from_multiple}개는 여러 옵션에서 추출되었습니다.")

def main():
    """메인 함수 - 모든 JSON 파일 처리"""
    # JSON 파일 목록
    json_files = glob.glob(os.path.join(BASE_DIR, "*.json"))
    
    if not json_files:
        print(f"{BASE_DIR}에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"처리할 JSON 파일 {len(json_files)}개를 찾았습니다.")
    
    # 기능 실행 안내
    if RUN_SIMILARITY and RUN_EXTRACTION:
        print("유사성 평가와 정답 추출을 모두 실행합니다.")
    elif RUN_SIMILARITY:
        print("유사성 평가만 실행합니다.")
    elif RUN_EXTRACTION:
        print("정답 추출만 실행합니다.")
    else:
        print("실행할 기능이 없습니다.")
        return
    
    # 각 파일 처리
    for json_file_path in json_files:
        process_json_file(json_file_path)
    
    print("모든 파일이 성공적으로 처리되었습니다.")

if __name__ == "__main__":
    main()