import os,sys
import json
import concurrent.futures
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

def load_gsm8k_dataset_from_huggingface() -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face datasets
    
    Returns:
        List of GSM8K questions with processed fields
    """
    # Load GSM8K dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")
    
    # Use the 'test' split for evaluation
    data = dataset["test"]
    
    formatted_questions = []
    for item in data:
        question_dict = {
            "question": item.get("question", ""),
            "answer": item.get("answer", "")
        }
        
        # Format the prompt for the model using the same format as SVAMP
        prompt = f"""
        Let's think step by step
        {question_dict["question"]}"""
        
        question_dict["prompt"] = prompt
        
        # Extract the original answer (extract the numerical value from the last sentence)
        answer_text = question_dict["answer"]
        # The original answer in GSM8K includes the reasoning steps, so let's extract the final numerical answer
        last_line = answer_text.strip().split('\n')[-1]
        if "answer is" in last_line.lower():
            original_answer = last_line.lower().split("answer is")[-1].strip()
            # Remove any additional text, periods, etc.
            original_answer = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', original_answer))
        else:
            # If "answer is" is not found, try to extract the last number from the text
            import re
            numbers = re.findall(r'\d+\.?\d*', last_line)
            original_answer = numbers[-1] if numbers else ""
            
        question_dict["original_answer"] = original_answer
        
        formatted_questions.append(question_dict)
    
    return formatted_questions

def skip_answered_questions(results_path: str, all_questions: list) -> list:
    if not os.path.exists(results_path):
        return all_questions

    with open(results_path, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)

    answered_questions_set = set(item["question"] for item in existing_results)
    remaining_questions = [q for q in all_questions if q["question"] not in answered_questions_set]
    return remaining_questions

def run_gsm8k_benchmark(
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9,
    parallel: bool = True
) -> str:

    all_questions = load_gsm8k_dataset_from_huggingface()
    results_filename = f"gsm8k_{model_name}.json"
    results_path = os.path.join(output_dir, results_filename)

    unanswered_questions = skip_answered_questions(results_path, all_questions)
    print(f"Loaded {len(all_questions)} questions ({len(unanswered_questions)} unanswered) from GSM8K")

    runner = LLMRunner(model_name)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

    for q_idx, question in enumerate(tqdm(unanswered_questions, desc=f"Processing {model_name}")):
        try:
            responses = runner.generate_responses(
                prompt=question["prompt"],
                num_responses=5,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=parallel
            )

            result_item = {
                "question": question["question"],
                "answer": question["answer"],
                "original_answer": question["original_answer"]
            }

            for i, response in enumerate(responses, 1):
                result_item[f"response_{i}"] = response

            all_results.append(result_item)

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error generating responses for question {q_idx+1}: {e}")
            continue

    print(f"Benchmark completed. Final results saved to {results_path}")
    return results_path


def run_all_models_parallel(
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> Dict[str, Optional[str]]:
    """
    Run benchmark for all available models in parallel
    
    Args:
        output_dir: Directory to save results
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary mapping model names to output file paths
    """
    available_models = list(LLMRunner.AVAILABLE_MODELS.keys())
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit jobs for each model
        future_to_model = {}
        for model_name in available_models:
            future = executor.submit(
                run_gsm8k_benchmark,
                model_name,
                output_dir,
                max_new_tokens,
                temperature,
                top_p,
                True  # Use parallel response generation
            )
            future_to_model[future] = model_name
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                output_path = future.result()
                results[model_name] = output_path
                print(f"Completed model: {model_name}")
            except Exception as e:
                print(f"Error running model {model_name}: {e}")
                results[model_name] = None
    
    return results


# Configuration parameters - modify these directly in the code
OUTPUT_DIR = "./Benchmarks/GSM8K/Results/zero_shot_gsm8k_results"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.5
TOP_P = 0.9
RUN_PARALLEL = True  # Set to False to run only one model

# Main execution
if __name__ == "__main__":
    if RUN_PARALLEL:
        # 모델을 두 그룹으로 나누기
        model_groups = [
            ["qwen-0.5b", "qwen-1.5b", "qwen-3b","deepseek-1.5b"],  # 첫 번째 그룹
            ["gemma-1b", "llama-1b", "llama-3b"]    # 두 번째 그룹
        ]
        
        all_results = {}
        
        # 각 그룹을 순차적으로 실행
        for group_idx, model_group in enumerate(model_groups):
            print(f"\n=== 실행 그룹 {group_idx+1}/{len(model_groups)}: {model_group} ===\n")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                # 그룹 내 각 모델에 대한 작업 제출
                future_to_model = {}
                for model_name in model_group:
                    future = executor.submit(
                        run_gsm8k_benchmark,
                        model_name,
                        OUTPUT_DIR,
                        MAX_NEW_TOKENS,
                        TEMPERATURE,
                        TOP_P,
                        True  # 병렬 응답 생성 사용
                    )
                    future_to_model[future] = model_name
                
                # 완료되는 대로 결과 수집
                group_results = {}
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        output_path = future.result()
                        group_results[model_name] = output_path
                        all_results[model_name] = output_path
                        print(f"Completed model: {model_name}")
                    except Exception as e:
                        print(f"Error running model {model_name}: {e}")
                        group_results[model_name] = None
                        all_results[model_name] = None
            
            # 각 그룹 완료 후 중간 결과 보고
            print(f"\n그룹 {group_idx+1} 벤치마크 완료. 결과 요약:")
            for model_name, output_path in group_results.items():
                if output_path:
                    print(f" {model_name}: {output_path}")
                else:
                    print(f" {model_name}: Failed")
        
        # 모든 그룹 완료 후 최종 결과 보고
        print("\n모든 그룹 벤치마크 완료. 전체 결과 요약:")
        for model_name, output_path in all_results.items():
            if output_path:
                print(f" {model_name}: {output_path}")
            else:
                print(f"{model_name}: Failed")
    else:
        # Run for a single model
        model_name = "qwen-0.5b"  # Change to desired model
        print(f"Running benchmark for model: {model_name}")
        output_path = run_gsm8k_benchmark(
            model_name=model_name,
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            parallel=True  # Can still use parallel response generation
        )
        print(f"Benchmark completed. Results saved to: {output_path}")