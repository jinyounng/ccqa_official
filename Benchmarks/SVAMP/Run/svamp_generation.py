import os
import json
import sys
import time
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
        
        # Format the prompt for the model using the specified format
        prompt = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {question_dict["body"]}{question_dict["question"]} A:"""
        
        question_dict["prompt"] = prompt
        formatted_questions.append(question_dict)
    
    return formatted_questions

def load_existing_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load existing results from a file if it exists
    
    Args:
        results_path: Path to the results file
        
    Returns:
        List of existing result items or empty list if file doesn't exist
    """
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different result formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "results" in data:
                return data["results"]
            else:
                return []
        except Exception as e:
            print(f"Error loading existing results from {results_path}: {e}")
            # Create a backup of the problematic file
            if os.path.exists(results_path):
                backup_path = f"{results_path}.bak.{int(time.time())}"
                try:
                    os.rename(results_path, backup_path)
                    print(f"Created backup of problematic file at {backup_path}")
                except Exception as be:
                    print(f"Failed to create backup: {be}")
            return []
    return []

def run_svamp_benchmark(
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
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
    
    # Check for existing results
    existing_results = load_existing_results(results_path)
    
    # Calculate how many questions have already been processed
    processed_count = len(existing_results)
    print(f"Found {processed_count} already processed questions for {model_name}")
    
    # Skip processed questions
    questions_to_process = questions[processed_count:]
    
    if not questions_to_process:
        print(f"All questions already processed for {model_name}. Nothing to do.")
        return results_path
    
    print(f"Will process {len(questions_to_process)} remaining questions for {model_name}")
    
    # Initialize model
    runner = LLMRunner(model_name)
    
    # Combine existing results with new ones
    all_results = existing_results.copy()
    
    # Process each question with tqdm progress bar
    for q_idx, question in enumerate(tqdm(questions_to_process, desc=f"Processing {model_name}")):
        global_idx = q_idx + processed_count
        
        # Generate responses
        try:
            responses = runner.generate_responses(
                prompt=question["prompt"],
                num_responses=20,  
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=parallel
            )
            
            # Create the result item with the requested format
            result_item = {
                "body": question["body"],
                "question": question["question"],
                "equation": question["equation"],
                "original_answer": question["original_answer"]
            }
            
            # Add responses with numbered keys
            for i, response in enumerate(responses, 1):
                result_item[f"response_{i}"] = response
            
            # Add to the complete results list
            all_results.append(result_item)
            
            # Save the complete results list after each item to allow resuming
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            # Print progress periodically
            if (global_idx + 1) % 10 == 0 or global_idx == 0 or global_idx == len(questions) - 1:
                print(f"Completed {global_idx + 1}/{len(questions)} questions ({((global_idx + 1) / len(questions) * 100):.1f}%)")
                
        except Exception as e:
            error_msg = f"Error generating responses for question {global_idx+1}: {e}"
            print(error_msg)
            
            # Save progress even when there's an error
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            continue
    
    print(f"Benchmark completed. Results saved to {results_path}")
    return results_path

def run_all_models_parallel(
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> Dict[str, Optional[str]]:
    
    available_models = list(LLMRunner.AVAILABLE_MODELS.keys())
    results = {}
    
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit jobs for each model
        future_to_model = {}
        for model_name in available_models:
            future = executor.submit(
                run_svamp_benchmark,
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


OUTPUT_DIR = "./Benchmarks/SVAMP/Results/svamp_numgeneration_20"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.5
TOP_P = 0.9
RUN_PARALLEL = True

# Main execution
if __name__ == "__main__":
    if RUN_PARALLEL:
        print("Starting parallel benchmark for all models...")
        results = run_all_models_parallel(
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        
        # Report results
        print("\nBenchmark completed. Results summary:")
        for model_name, output_path in results.items():
            if output_path:
                print(f"SUCCESS: {model_name}: {output_path}")
            else:
                print(f"FAILED: {model_name}")
    else:
        # Run for a single model
        model_name = "gemma-1b"  # Change to desired model
        print(f"Running benchmark for model: {model_name}")
        output_path = run_svamp_benchmark(
            model_name=model_name,
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            parallel=True  
        )
        print(f"Benchmark completed. Results saved to: {output_path}")