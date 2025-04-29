import json
import os
import sys
from typing import Dict, List, Any
from tqdm import tqdm
from datasets import load_dataset
import torch
import multiprocessing as mp

sys.path.append('/data3/jykim/Projects/CCQA_official')
# LLM_runner 모듈 임포트
from LLM_runner import LLMRunner

# Set start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# StrategyQA prompt examples
STRATEGY_QA_PROMPT = """Q: Do hamsters provide food for any animals? A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.
Q: Could Brooke Shields succeed at University of Pennsylvania? A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.
Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls? A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no.
Q: Yes or no: Is it common to see frost during some college commencements? A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.
Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.
Q: Yes or no: Would a pear sink in water? A: The density of a pear is about 0.6g/cm3 , which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no."""

def format_model_name(model_name):
    """Format model name to proper case (e.g., 'llama-3b' -> 'Llama-3B')"""
    parts = model_name.split('-')
    formatted_parts = []
    
    for part in parts:
        # Check if the part has digits at the end (like "3b")
        if any(c.isdigit() for c in part):
            # Find where digits start
            for i, c in enumerate(part):
                if c.isdigit():
                    digit_start = i
                    break
            else:
                digit_start = len(part)
            
            # Capitalize the letter part and uppercase the size suffix
            letter_part = part[:digit_start].capitalize()
            size_suffix = part[digit_start:].upper()
            formatted_parts.append(f"{letter_part}{size_suffix}")
        else:
            # Just capitalize regular parts
            formatted_parts.append(part.capitalize())
    
    return '-'.join(formatted_parts)

def get_all_possible_model_filenames(model_name, output_dir):
    """
    다른 가능한 이름 형식으로 파일 경로를 생성합니다.
    예: llama-3b → [Llama-3B, llama-3b, Llama-3.2-3B-Instruct]
    """
    possible_names = []
    
    # 1. 원래 모델 이름으로 된 파일명
    possible_names.append(os.path.join(output_dir, f"strategyqa_{model_name}_result.json"))
    
    # 2. 포맷팅된 모델 이름으로 된 파일명
    formatted_name = format_model_name(model_name)
    possible_names.append(os.path.join(output_dir, f"strategyqa_{formatted_name}_result.json"))
    
    # 3. Llama-3.2-3B-Instruct 형식 확인 (llama-3b와 같은 경우)
    if model_name == "llama-3b":
        possible_names.append(os.path.join(output_dir, f"strategyqa_Llama-3.2-3B-Instruct_result.json"))
    elif model_name == "llama-7b":
        possible_names.append(os.path.join(output_dir, f"strategyqa_Llama-3.2-7B-Instruct_result.json"))
    
    return possible_names

def run_benchmark(args):
    """Run the benchmark for a specific model."""
    model_name, dataset, output_dir = args
    print(f"\nRunning benchmark for {model_name}...")
    
    # LLMRunner 초기화
    llm_runner = LLMRunner(model_name=model_name)
    
    # 모델 이름 형식 변경 (파일명용)
    formatted_model_name = format_model_name(model_name)
    
    # 결과 파일 경로 (새로운 형식)
    output_file = os.path.join(output_dir, f"strategyqa_{formatted_model_name}_result.json")
    
    # 가능한 모든 파일명 검사 (기존 파일과의 호환성)
    possible_files = get_all_possible_model_filenames(model_name, output_dir)
    existing_file = None
    
    # 기존 파일 검색
    for file_path in possible_files:
        if os.path.exists(file_path):
            existing_file = file_path
            print(f"Found existing result file: {os.path.basename(existing_file)}")
            break
    
    # 기존 결과 파일이 있다면 로드
    if existing_file:
        with open(existing_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # 이미 처리된 문항들의 ID를 저장
        processed_ids = {result['qid'] for result in results}
        # 찾은 파일을 출력 파일로 사용 (기존 파일명 유지)
        output_file = existing_file
    else:
        results = []
        processed_ids = set()
    
    # 처리된 문제 수 출력
    print(f"Dataset size: {len(dataset)}")
    print(f"Current results size: {len(results)}")
    print(f"Using result file: {os.path.basename(output_file)}")
    
    for item in tqdm(dataset, desc=f"Processing {model_name}"):
        # 이미 처리된 항목은 건너뛰기
        if item['qid'] in processed_ids:
            continue
            
        try:
            # StrategyQA 형식은 단순히 질문과 예/아니오 답변이므로 선택지 포맷팅이 필요 없음
            question = item['question']
            
            # 프롬프트 생성 - 'Yes or no:' 접두어 추가
            if not question.startswith("Yes or no:"):
                prompt_question = f"Yes or no: {question}"
            else:
                prompt_question = question
                
            # 전체 프롬프트 생성
            prompt = f"{STRATEGY_QA_PROMPT}\nQ: {prompt_question}\n A:"
            
            # 20개 응답 생성
            responses = llm_runner.generate_responses(
                prompt=prompt,
                num_responses=20,  # 20개 응답 생성
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                parallel=True  # 병렬 실행 활성화 (여러 응답 생성 속도 향상)
            )
            
            # 결과 항목 생성 (개별 응답을 response_1, response_2, ... 형식으로 저장)
            result_item = {
                'qid': item['qid'],
                'question': item['question'],
                'answer': item['answer'],  # StrategyQA는 True/False 값을 가짐
            }
            
            # responses를 각각의 개별 항목으로 저장
            for i, response in enumerate(responses, 1):
                result_item[f'response_{i}'] = response
            
        except Exception as e:
            print(f"Error processing question: {item['question']}")
            print(f"Error: {str(e)}")
            result_item = {
                'qid': item.get('qid', ''),
                'question': item['question'],
                'answer': item.get('answer', False),
                'error': str(e)
            }
            # 에러 발생 시에도 빈 response 항목들 생성
            for i in range(1, 2):  # 1개의 빈 응답 항목 생성
                result_item[f'response_{i}'] = 'ERROR'
        
        # 결과를 리스트에 추가
        results.append(result_item)
        processed_ids.add(item['qid'])
        
        # 결과를 즉시 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return model_name, len(results)

def process_model_group(model_group, dataset, output_dir):
    """Process a group of models in parallel."""
    pool = mp.Pool(processes=len(model_group))
    try:
        args = [(model_name, dataset, output_dir) for model_name in model_group]
        results = pool.map(run_benchmark, args)
    finally:
        pool.close()
        pool.join()
        del pool  # 명시적으로 pool 객체 삭제
    return dict(results)

def main(test_sample: bool = False):
    # Create output directory if it doesn't exist
    output_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/StrategyQA/Results/strategyqa_result"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the StrategyQA dataset
    print(f"Loading StrategyQA dataset...")
    dataset = load_dataset("ChilleD/StrategyQA")
    validation_data = dataset["test"]
    
    # 테스트 모드일 경우 첫 번째 샘플만 선택
    if test_sample:
        validation_data = validation_data.select(range(1))
        print(f"\nTesting with single sample:")
        sample = validation_data[0]
        print(f"\nQuestion: {sample['question']}")
        print(f"\nAnswer: {sample['answer']}")
    
    # 사용할 모델 목록 (LLMRunner.AVAILABLE_MODELS의 키 사용)
    model_names = [
        "llama-1b",
        "llama-3b",
        "qwen-0.5b",
        "qwen-1.5b",
        "qwen-3b",
        "deepseek-1.5b",
        "falcon-1b"
    ]
    
    for model in model_names:
        print(f"Model name: {model} -> {format_model_name(model)}")
    
    group_size = 7  # GPU 수 또는 사용 가능한 리소스에 맞게 조정
    model_groups = [model_names[i:i+group_size] for i in range(0, len(model_names), group_size)]
    
    # Process each group of models
    for group in model_groups:
        print(f"\nProcessing model group: {group}")
        process_model_group(group, validation_data, output_dir)
    
    print(f"\nBenchmark complete.")

if __name__ == "__main__":
    try:
        main(test_sample=False)
    finally:
        # 프로그램 종료 시 리소스 정리
        if mp.current_process().name == 'MainProcess':
            mp.active_children()  # 활성 자식 프로세스 정리
            
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()