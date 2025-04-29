import os
import sys
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import glob

# 원래 코드에서 LLMRunner를 사용하기 위해 경로 추가
sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

def format_prompt_for_question(question: str) -> str:
    """
    GSM8K 형식으로 질문에 대한 프롬프트 생성
    
    Args:
        question: 질문 텍스트
        
    Returns:
        모델에 입력할 프롬프트
    """
    # GSM8K 스타일 프롬프트 형식
    prompt = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {question} A:"""
    
    return prompt

def map_filename_to_model_name(model_name: str) -> str:
    """
    파일 이름에서 추출한 모델 이름을 LLMRunner가 지원하는 모델 이름으로 매핑
    
    Args:
        model_name: 파일 이름에서 추출한 모델 이름
        
    Returns:
        LLMRunner가 지원하는 모델 이름
    """
    # 모델 이름 매핑 테이블
    model_mapping = {
        "Qwen2.5-1.5B-Instruct": "qwen-1.5b",
        "Qwen2.5-0.5B-Instruct": "qwen-0.5b", 
        "Qwen2.5-3B-Instruct": "qwen-3b",
        "Deepseek-1.3B-Chat": "deepseek-1.5b",
        "LLaMA3.1-1B-Instruct": "llama-1b",
        "LLaMA3.1-3B-Instruct": "llama-3b",
        "Falcon3-1B-Instruct": "falcon-1b"
    }
    
    return model_mapping.get(model_name, model_name)

def extract_model_name_from_filename(filename: str) -> str:
    """
    파일 이름에서 모델 이름 추출하고 LLMRunner가 지원하는 이름으로 매핑
    
    Args:
        filename: JSON 파일 이름
        
    Returns:
        LLMRunner가 지원하는 모델 이름
    """
    base_name = os.path.basename(filename)
    # 파일명 형식: GSM8K_[모델명]_few_shot_result_self_consistency.json
    parts = base_name.split('_')
    if len(parts) >= 2:
        extracted_name = parts[1]  # 일반적으로 모델명은 두 번째 부분
        return map_filename_to_model_name(extracted_name)
    return "unknown-model"

def process_file(file_path: str, 
                 num_responses: int = 5, 
                 max_new_tokens: int = 2048, 
                 temperature: float = 0.5, 
                 top_p: float = 0.9,
                 parallel: bool = True) -> None:
    """
    JSON 파일 처리 및 생성된 질문에 대한 답변 생성
    
    Args:
        file_path: JSON 파일 경로
        num_responses: 생성할 응답 수
        max_new_tokens: 최대 토큰 수
        temperature: 생성 온도
        top_p: top-p 샘플링 파라미터
        parallel: 병렬 응답 생성 여부
    """
    print(f"파일 처리 중: {file_path}")
    
    # 경과 시간 측정 시작
    start_time = time.time()
    
    # JSON 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 파일 이름에서 원래 모델 이름 추출
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    original_model_name = parts[1] if len(parts) >= 2 else "unknown"
    print(f"파일에서 추출한 원래 모델 이름: {original_model_name}")
    
    # LLMRunner용 모델 이름으로 매핑
    model_name = extract_model_name_from_filename(file_path)
    print(f"LLMRunner용으로 매핑된 모델 이름: {model_name}")
    
    # LLM 실행기 초기화
    try:
        runner = LLMRunner(model_name)
        print(f"LLMRunner 초기화 성공: {model_name}")
    except Exception as e:
        print(f"LLMRunner 초기화 실패: {e}")
        return
    
    # 결과 항목 수 확인
    results_count = len(data.get("results", []))
    print(f"총 문항 수: {results_count}")
    
    # 진행 상황을 위한 카운터
    processed_count = 0
    total_questions = results_count * 5  # 각 문항당 5개의 질문 가능
    
    # 각 문항에 대해 처리
    for idx, item in enumerate(data.get("results", [])):
        # 각 생성된 질문(1-5)에 대해 처리
        for q_idx in range(1, 6):
            gen_question_key = f"generated_question_{q_idx}"
            gen_answer_key = f"generated_answer_{q_idx}"
            
            # 진행률 표시를 위한 카운터 증가
            processed_count += 1
            progress_percent = (processed_count / total_questions) * 100
            print(f"[{model_name}] 진행률: {progress_percent:.2f}% ({processed_count}/{total_questions})")
            
            # 이미 응답이 있다면 건너뛰기
            if any(f"{gen_answer_key}_{resp_idx}" in item for resp_idx in range(1, num_responses+1)):
                print(f"이미 답변이 있음: {gen_answer_key}")
                continue
            
            # 질문이 있는지 확인
            if gen_question_key in item and item[gen_question_key]:
                question = item[gen_question_key]
                prompt = format_prompt_for_question(question)
                
                try:
                    print(f"질문 '{gen_question_key}' 응답 생성 중...")
                    # 모델로부터 응답 생성
                    responses = runner.generate_responses(
                        prompt=prompt,
                        num_responses=num_responses,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        parallel=parallel
                    )
                    
                    # 응답 저장
                    for resp_idx, response in enumerate(responses, 1):
                        item[f"{gen_answer_key}_{resp_idx}"] = response
                    
                    print(f"'{gen_question_key}' 응답 생성 완료 ({len(responses)}개)")
                    
                    # 중간 결과 저장
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"응답 생성 중 오류 발생: {e}")
            else:
                if gen_question_key not in item:
                    print(f"질문 '{gen_question_key}'가 없음")
                elif not item[gen_question_key]:
                    print(f"질문 '{gen_question_key}'가 비어 있음")
    
    # 최종 결과 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 경과 시간 계산 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    
    print(f"파일 처리 완료: {file_path}")
    print(f"총 처리 시간: {elapsed_time:.2f}초 ({elapsed_minutes:.2f}분)")
    
    # 처리된 질문 수 계산
    processed_questions = 0
    for item in data.get("results", []):
        for q_idx in range(1, 6):
            gen_question_key = f"generated_question_{q_idx}"
            gen_answer_key = f"generated_answer_{q_idx}"
            if gen_question_key in item and any(f"{gen_answer_key}_{resp_idx}" in item for resp_idx in range(1, num_responses+1)):
                processed_questions += 1
    
    print(f"처리된 질문 수: {processed_questions}")
    if processed_questions > 0:
        print(f"질문당 평균 처리 시간: {elapsed_time/processed_questions:.2f}초")

def main():
    # 설정 파라미터
    RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_0_result"
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.5
    TOP_P = 0.9
    NUM_RESPONSES = 1
    PARALLEL_RESPONSES = True  # 응답 병렬 생성 (LLMRunner 내부)
    PARALLEL_MODELS = True  # 모델(파일) 병렬 처리
    MAX_WORKERS = 6  # 최대 동시 처리 모델 수
    
    # 결과 디렉토리 존재 확인
    if not os.path.exists(RESULTS_DIR):
        print(f"결과 디렉토리가 존재하지 않습니다: {RESULTS_DIR}")
        return
    
    # 모든 JSON 파일 가져오기
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    print(f"찾은 JSON 파일 수: {len(json_files)}")
    
    if PARALLEL_MODELS:
        # 병렬로 여러 모델(파일) 처리
        print(f"모델 병렬 처리 시작 (최대 {MAX_WORKERS}개 동시 처리)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for file_path in json_files:
                future = executor.submit(
                    process_file,
                    file_path=file_path,
                    num_responses=NUM_RESPONSES,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    parallel=PARALLEL_RESPONSES
                )
                futures[future] = os.path.basename(file_path)
            
            # 결과 수집
            for future in concurrent.futures.as_completed(futures):
                file_name = futures[future]
                try:
                    future.result()  # 결과 받기 (process_file에서는 반환값 없음)
                    print(f"모델 병렬 처리 완료: {file_name}")
                except Exception as e:
                    print(f"모델 병렬 처리 중 오류 발생: {file_name}, 오류: {e}")
    else:
        # 순차적으로 파일 처리
        for file_path in json_files:
            try:
                print(f"\n=== 파일 처리 시작: {file_path} ===\n")
                process_file(
                    file_path=file_path,
                    num_responses=NUM_RESPONSES,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    parallel=PARALLEL_RESPONSES
                )
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")

if __name__ == "__main__":
    main()