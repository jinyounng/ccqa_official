import os,sys
import json
import concurrent.futures
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

def load_svamp_dataset_from_huggingface() -> List[Dict[str, Any]]:
    """
    Load SVAMP dataset from Hugging Face datasets
    
    Returns:
        List of SVAMP questions with processed fields
    """
    # Load SVAMP dataset from Hugging Face
    dataset = load_dataset("ChilleD/SVAMP")
    
    # Use the 'test' split as it contains all the data
    data = dataset["test"]
    
    formatted_questions = []
    for item in data:
        question_dict = {
            "body": item.get("Body", ""),
            "question": item.get("Question", ""),
            "equation": item.get("Equation", ""),
            "original_answer": str(item.get("Answer", ""))
        }
        
        # Format the prompt with zero-shot "Let's think step by step" instruction
        prompt = f"""Q: {question_dict["body"]}{question_dict["question"]}

Let's think step by step.
"""
        
        question_dict["prompt"] = prompt
        formatted_questions.append(question_dict)
    
    return formatted_questions

def run_svamp_benchmark(
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.5,
    top_p: float = 0.9,
    parallel: bool = True
) -> str:
    # Load dataset from Hugging Face
    questions = load_svamp_dataset_from_huggingface()
    print(f"Loaded {len(questions)} questions from SVAMP dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a results file
    results_filename = f"svamp_{model_name}.json"
    results_path = os.path.join(output_dir, results_filename)
    
    # 이미 존재하는 결과 파일 확인 및 로드
    all_results = []
    processed_questions = set()
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            
            # 이미 처리된 문제 추적
            for item in all_results:
                question_key = f"{item.get('body', '')}{item.get('question', '')}"
                processed_questions.add(question_key)
            
            print(f"이미 처리된 {len(all_results)}개 결과를 {results_path}에서 로드했습니다.")
        except Exception as e:
            print(f"기존 결과 로딩 오류: {e}")
            # 오류 발생 시 빈 결과 목록으로 시작
            all_results = []
    
    # 모델 초기화 (이미 처리된 문제가 있는 경우에만)
    runner = None
    if len(processed_questions) < len(questions):
        runner = LLMRunner(model_name)
    
    # 각 문제 처리 (tqdm 진행 표시줄 사용)
    for q_idx, question in enumerate(tqdm(questions, desc=f"Processing {model_name}")):
        # 이미 처리된 문제는 건너뛰기
        question_key = f"{question['body']}{question['question']}"
        if question_key in processed_questions:
            continue
        
        # 모델이 초기화되지 않은 경우 지금 초기화
        if runner is None:
            runner = LLMRunner(model_name)
        
        # 응답 생성
        try:
            responses = runner.generate_responses(
                prompt=question["prompt"],
                num_responses=20,  
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=parallel
            )
            
            # 요청된 형식으로 결과 항목 생성
            result_item = {
                "body": question["body"],
                "question": question["question"],
                "equation": question["equation"],
                "original_answer": question["original_answer"]
            }
            
            # 번호가 매겨진 키로 응답 추가
            for i, response in enumerate(responses, 1):
                result_item[f"response_{i}"] = response
            
            # 전체 결과 목록에 추가
            all_results.append(result_item)
            processed_questions.add(question_key)
            
            # 각 항목 후 전체 결과 목록 저장
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"질문 {q_idx+1}에 대한 응답 생성 오류: {e}"
            print(error_msg)
            continue
    
    print(f"벤치마크 완료. 최종 결과가 {results_path}에 저장되었습니다.")
    return results_path

def run_all_models_parallel(
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_parallel_models: int = 2  # 동시에 실행할 모델 수 제한
) -> Dict[str, Optional[str]]:
    
    available_models = list(LLMRunner.AVAILABLE_MODELS.keys())
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
        # 각 모델에 대한 작업 제출
        future_to_model = {}
        for model_name in available_models:
            future = executor.submit(
                run_svamp_benchmark,
                model_name,
                output_dir,
                max_new_tokens,
                temperature,
                top_p,
                True  # 병렬 응답 생성 사용
            )
            future_to_model[future] = model_name
        
        # 완료되는 대로 결과 수집
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                output_path = future.result()
                results[model_name] = output_path
                print(f"모델 완료: {model_name}")
            except Exception as e:
                print(f"모델 {model_name} 실행 오류: {e}")
                results[model_name] = None
    
    return results


OUTPUT_DIR = "./Benchmarks/SVAMP/Results/svamp_zeroshot_results"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.5
TOP_P = 0.9
RUN_PARALLEL = True # 단일 모델만 실행하려면 False로 설정
if __name__ == "__main__":
    if RUN_PARALLEL:
        print("모든 모델에 대해 zero-shot 'Let's think step by step' 프롬프트로 병렬 벤치마크 시작...")
        print("한 번에 최대 2개 모델만 병렬 실행됩니다.")
        results = run_all_models_parallel(
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_parallel_models=1  # 최대 3개 모델 동시 실행
        )
        
        # 결과 보고
        print("\n벤치마크 완료. 결과 요약:")
        for model_name, output_path in results.items():
            if output_path:
                print(f" {model_name}: {output_path}")
            else:
                print(f" {model_name}: 실패")
    else:
        # 단일 모델 실행 (기존 코드 유지)
        model_name = "llama-3b"  # 원하는 모델로 변경
        print(f"모델 {model_name}에 대해 zero-shot 'Let's think step by step' 프롬프트로 벤치마크 실행")
        output_path = run_svamp_benchmark(
            model_name=model_name,
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            parallel=True  # 병렬 응답 생성은 계속 사용 가능
        )
        print(f"벤치마크 완료. 결과가 {output_path}에 저장되었습니다.")