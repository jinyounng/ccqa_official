import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import concurrent.futures
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import multiprocessing as mp
# Add project root path to Python path
sys.path.append('/data3/jykim/Projects/CCQA_official')

class T5QuestionGenerator:
    """T5 model for generating questions from answers"""
    
    def __init__(self, model_path: str, device: str = None):
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading T5 model from '{model_path}'... Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        
        print(f"T5 model loaded successfully")
    
    def generate_question(self, body: str, answer: str, max_length: int = 128) -> str:
        # Format input for T5 with the detailed prompt template
        input_text = f"""CRITICAL: Do not change ANY numeric values in the answer. 
        Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. 
        Generate a question that would have this as its answer: {answer}"""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate question using settings from the test script
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=False,
            )
        
        # Decode question
        question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return question

def load_svamp_results(results_path: str) -> List[Dict[str, Any]]:
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Handle different results formats
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        print(f"Error loading results {results_path}: {e}")
        return []

def load_existing_ccqa_results(output_path: str) -> Dict[str, Any]:
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading existing CCQA results from {output_path}: {e}")
            
            # Create backup of problematic file
            backup_path = f"{output_path}.bak.{int(time.time())}"
            try:
                os.rename(output_path, backup_path)
                print(f"Created backup of problematic file at {backup_path}")
            except Exception as be:
                print(f"Failed to create backup: {be}")
    
    # Return empty structure
    return {
        "completed_questions": 0,
        "total_questions": 0,
        "results": []
    }

def generate_question_from_answer(
    model: T5QuestionGenerator,
    body: str,
    answer: str,
    max_length: int = 50
) -> str:
    try:
        return model.generate_question(body, answer, max_length=max_length)
    except Exception as e:
        print(f"Error generating question from answer: {e}")
        return f"Error: {str(e)}"

def apply_ccqa(
    t5_model: T5QuestionGenerator,
    results: List[Dict[str, Any]],
    output_path: str,
    max_length: int = 128
) -> List[Dict[str, Any]]:

    # Load existing CCQA results
    existing_data = load_existing_ccqa_results(output_path)
    ccqa_results = existing_data.get("results", [])
    
    # Calculate how many questions have already been processed
    processed_count = len(ccqa_results)
    total_questions = len(results)
    
    print(f"Found {processed_count} already processed questions")
    
    # Skip already processed questions
    results_to_process = results[processed_count:]
    
    if not results_to_process:
        print(f"All questions already processed. Nothing to do.")
        return ccqa_results
    
    print(f"Will process {len(results_to_process)} remaining questions")
    
    for i, item in enumerate(tqdm(results_to_process, desc="Applying CCQA")):
        global_idx = i + processed_count
        ccqa_item = item.copy()
        
        # Add which model was used for question generation
        ccqa_item["used_t5_model"] = True
        
        # Extract original question
        original_body = item.get("body", "")
        
        # Process each response
        for resp_idx in range(1, 21):  # Process 20 responses
            response_key = f"response_{resp_idx}"
            answer = item.get(response_key, "")
            
            if not answer:
                print(f"Warning: Could not find {response_key}. Question index: {global_idx}")
                continue
            
            try:
                # Generate a question from the answer
                generated_question = generate_question_from_answer(
                    model=t5_model,
                    body=original_body,
                    answer=answer,
                    max_length=max_length
                )
                
                # Store generated question
                ccqa_item[f"generated_question_{resp_idx}"] = generated_question
                
            except Exception as e:
                print(f"Error processing {response_key}: {e}")
                ccqa_item[f"generated_question_{resp_idx}"] = f"Error: {str(e)}"
        
        ccqa_results.append(ccqa_item)
        
        # Save progress after each question
        current_data = {
            "completed_questions": global_idx + 1,
            "total_questions": total_questions,
            "model_used": "t5",
            "results": ccqa_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        # Print progress periodically
        if (global_idx + 1) % 10 == 0 or global_idx == 0 or global_idx == total_questions - 1:
            print(f"Progress: {global_idx + 1}/{total_questions} questions completed ({((global_idx + 1) / total_questions * 100):.1f}%)")
    
    return ccqa_results

def run_ccqa_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen",
    max_length: int = 50
) -> Optional[Dict]:

    try:
        start_time = time.time()
        
        # Find this model's results file
        results_filename = f"svamp_{model_name}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        if not os.path.exists(results_path):
            print(f"Could not find results file for model {model_name}: {results_path}")
            return None
        
        # Load results
        results_data = load_svamp_results(results_path)
            
        if not results_data:
            print(f"No results found in {results_path}")
            return None
        
        print(f"Loaded {len(results_data)} results for model {model_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to save CCQA results
        ccqa_filename = f"svamp_{model_name}_ccqa_t5.json"
        ccqa_path = os.path.join(output_dir, ccqa_filename)
        
        # Initialize T5 model
        t5_model = T5QuestionGenerator(model_path=t5_model_path)
        
        # Apply CCQA (saving progress after each question)
        ccqa_results = apply_ccqa(
            t5_model=t5_model,
            results=results_data,
            output_path=ccqa_path,
            max_length=max_length
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time information
        time_info = {
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "processing_time_hours": processing_time / 3600,
            "questions_count": len(results_data),
            "avg_time_per_question": processing_time / len(results_data) if len(results_data) > 0 else 0,
            "used_t5_model": True
        }
        
        # Save final results with time information
        final_data = {
            "time_info": time_info,
            "results": ccqa_results
        }
        
        with open(ccqa_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"CCQA results (using T5) saved to {ccqa_path}")
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        return {
            "path": ccqa_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"Error running CCQA for model {model_name}: {e}")
        return None

def run_all_models_ccqa(
    results_dir: str,
    output_dir: str,
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen",
    max_length: int = 50,
    parallel: bool = True
) -> Dict[str, Dict]:

    # Find all result files in the directory
    result_files = [f for f in os.listdir(results_dir) if f.startswith("svamp_") and f.endswith(".json")]
    model_names = [f.replace("svamp_", "").replace(".json", "") for f in result_files]
    
    results = {}
    
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit jobs for each model
            future_to_model = {}
            for model_name in model_names:
                future = executor.submit(
                    run_ccqa_for_model,
                    model_name,
                    results_dir,
                    output_dir,
                    t5_model_path,
                    max_length
                )
                future_to_model[future] = model_name
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_info = future.result()
                    results[model_name] = result_info
                    if result_info:
                        time_info = result_info["time_info"]
                        print(f"CCQA (using T5) completed for model {model_name} (processing time: {time_info['processing_time_minutes']:.2f} minutes)")
                    else:
                        print(f"CCQA failed for model {model_name}")
                except Exception as e:
                    print(f"Error running CCQA for model {model_name}: {e}")
                    results[model_name] = None
    else:
        # Run sequentially
        for model_name in model_names:
            try:
                result_info = run_ccqa_for_model(
                    model_name=model_name,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    t5_model_path=t5_model_path,
                    max_length=max_length
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"Error running CCQA for model {model_name}: {e}")
                results[model_name] = None
    
    return results

def run_svamp_ccqa(
    results_dir: str,
    output_dir: str,
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen",
    max_length: int = 50,
    parallel: bool = False
) -> Dict[str, Dict]:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting CCQA (using T5) for SVAMP benchmark...")
    results = run_all_models_ccqa(
        results_dir=results_dir,
        output_dir=output_dir,
        t5_model_path=t5_model_path,
        max_length=max_length,
        parallel=parallel
    )
    
    # Report results
    print(f"\nCCQA (using T5) completed. Results summary:")
    
    # Save all model processing time information to a single JSON file
    time_summary = {}
    
    for model_name, result_info in results.items():
        if result_info:
            time_info = result_info["time_info"]
            path = result_info["path"]
            time_summary[model_name] = time_info
            print(f"SUCCESS: {model_name}:")
            print(f"   - Processing time: {time_info['processing_time_seconds']:.2f} seconds ({time_info['processing_time_minutes']:.2f} minutes)")
            print(f"   - Average time per question: {time_info['avg_time_per_question']:.2f} seconds")
            print(f"   - Output path: {path}")
        else:
            print(f"FAILED: {model_name}")
    
    # Save time information summary
    time_summary_path = os.path.join(output_dir, "ccqa_time_summary_t5.json")
    with open(time_summary_path, 'w', encoding='utf-8') as f:
        json.dump(time_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll processing completed. Results saved to {output_dir}")
    print(f"Time summary information saved to {time_summary_path}")
    
    return results

# Configuration parameters - modify these directly in the script
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/svamp_results"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result"
T5_MODEL_PATH = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen"
MAX_LENGTH = 50
RUN_PARALLEL = False  # Set to True to run models in parallel

# Main execution
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_svamp_ccqa(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        t5_model_path=T5_MODEL_PATH,
        max_length=MAX_LENGTH,
        parallel=RUN_PARALLEL
    )