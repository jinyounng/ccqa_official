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
        prompt = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {question_dict["question"]} A:"""
        
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
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
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
                num_responses=20,
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
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
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
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/gsm8k_numgeneration_20"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
RUN_PARALLEL = True  # Set to False to run only one model

# Main execution
if __name__ == "__main__":
    if RUN_PARALLEL:
        # 모델을 두 그룹으로 나누기
        model_groups = [
            ["qwen-0.5b", "qwen-1.5b", "qwen-3b","falcon-1b", "llama-1b", "llama-3b"] # 첫 번째 그룹
        ]
        
        all_results = {}
        
        # 각 그룹을 순차적으로 실행
        for group_idx, model_group in enumerate(model_groups):
            print(f"\n=== 실행 그룹 {group_idx+1}/{len(model_groups)}: {model_group} ===\n")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
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