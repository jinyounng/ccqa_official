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

# CSQA prompt examples
CSQA_PROMPT = """Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket (b) calligrapher's hand (c) inkwell (d) desk drawer (e) blotter A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).
Q: What home entertainment equipment requires cable? Answer Choices: (a) radio shack (b) substation (c) television (d) cabinet A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).
Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (c).
Q: Sammy wanted to go to where the people were. Where might he go? Answer Choices: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).
Q: Where do you put your grapes just before checking out? Answer Choices: (a) mouth (b) grocery cart (c)super market (d) fruit basket (e) fruit market A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).
Q: Google Maps and other highway and street GPS services have replaced what? Answer Choices: (a) united states (b) mexico (c) countryside (d) atlas A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).
Q: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c)."""

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
    possible_names.append(os.path.join(output_dir, f"CommonSenseQA_{model_name}_result.json"))
    
    # 2. 포맷팅된 모델 이름으로 된 파일명
    formatted_name = format_model_name(model_name)
    possible_names.append(os.path.join(output_dir, f"CommonSenseQA_{formatted_name}_result.json"))
    
    # 3. Llama-3.2-3B-Instruct 형식 확인 (llama-3b와 같은 경우)
    if model_name == "llama-3b":
        possible_names.append(os.path.join(output_dir, f"CommonSenseQA_Llama-3.2-3B-Instruct_result.json"))
    elif model_name == "llama-7b":
        possible_names.append(os.path.join(output_dir, f"CommonSenseQA_Llama-3.2-7B-Instruct_result.json"))
    
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
    output_file = os.path.join(output_dir, f"CommonSenseQA_{formatted_model_name}_result.json")
    
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
        processed_ids = {result['id'] for result in results}
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
        if item['id'] in processed_ids:
            continue
            
        try:
            # 선택지 목록 생성
            choices = [item["choices"]["text"][i] for i in range(len(item["choices"]["text"]))]
            choices_formatted = ", ".join([f"{chr(65+i)}: {choices[i]}" for i in range(len(choices))])
            
            # 질문과 선택지 결합
            question_with_choices = f"{item['question']}\nChoices: {choices_formatted}"
            
            # 프롬프트 생성
            prompt = f"{CSQA_PROMPT}\n\nQ: {question_with_choices}A:"
            
            # 5개 응답 생성
            responses = llm_runner.generate_responses(
                prompt=prompt,
                num_responses=5,  # 5개 응답 생성
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                parallel=True  # 병렬 실행 활성화 (여러 응답 생성 속도 향상)
            )
            
            # 결과 항목 생성 (개별 응답 항목으로 저장)
            result_item = {
                'id': item['id'],
                'question': item['question'],
                'choices': choices,
                'answerKey': item.get('answerKey', '')
            }
            
            # 개별 응답 항목으로 추가
            for i, response in enumerate(responses, 1):
                result_item[f'response_{i}'] = response
            
        except Exception as e:
            print(f"Error processing question: {item['question']}")
            print(f"Error: {str(e)}")
            result_item = {
                'id': item.get('id', ''),
                'question': item['question'],
                'choices': item.get('choices', {}).get('text', []),
                'answerKey': item.get('answerKey', ''),
                'response_1': 'ERROR',
                'response_2': 'ERROR',
                'response_3': 'ERROR',
                'response_4': 'ERROR',
                'response_5': 'ERROR',
                'error': str(e)
            }
        
        # 결과를 리스트에 추가
        results.append(result_item)
        processed_ids.add(item['id'])
        
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
    output_dir = "/data3/jykim/Projects/CCQA_official/finetuning/train_set/commonsenseqa_train"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CommonsenseQA dataset (validation split)
    print(f"Loading CommonsenseQA dataset (train split)...")
    dataset = load_dataset("commonsense_qa")
    validation_data = dataset["train"]
    
    # 테스트 모드일 경우 첫 번째 샘플만 선택
    if test_sample:
        validation_data = validation_data.select(range(1))
        print(f"\nTesting with single sample:")
        sample = validation_data[0]
        print(f"\nQuestion: {sample['question']}")
        print(f"\nChoices: {sample['choices']['text']}")
        print(f"\nAnswer: {sample['answerKey']}")
    
    # 사용할 모델 목록 (LLMRunner.AVAILABLE_MODELS의 키 사용)
    model_names = [
        "llama-1b",
        "llama-3b",
        "deepseek-1.5b",
        "qwen-0.5b",
        "qwen-1.5b",
        "qwen-3b",
    ]
    
    # 개선된 모델 이름 출력 (참고용)
    for model in model_names:
        print(f"Model name: {model} -> {format_model_name(model)}")
    
    group_size = 6 # GPU 수 또는 사용 가능한 리소스에 맞게 조정
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