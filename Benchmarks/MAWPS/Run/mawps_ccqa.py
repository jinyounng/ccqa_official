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
                temperature=0.0,
                no_repeat_ngram_size=2
            )
        
        # Decode question
        question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return question

def load_mawps_results(results_path: str) -> List[Dict[str, Any]]:
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # MAWPS 데이터는 리스트 형태로 저장됨
        if isinstance(results, list):
            return results
        elif isinstance(results, dict) and "results" in results:
            return results["results"]  # 호환성을 위해 유지
        else:
            print(f"Unexpected format in {results_path}")
            return []
    except Exception as e:
        print(f"Error loading results {results_path}: {e}")
        return []

def load_existing_ccqa_results(output_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # MAWPS 결과는 리스트 형태로 저장됨
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "results" in data:
                return data["results"]  # 호환성을 위해 유지
            else:
                print(f"Unexpected format in existing CCQA results: {output_path}")
                return []
        except Exception as e:
            print(f"Error loading existing CCQA results from {output_path}: {e}")
            
            # Create backup of problematic file
            backup_path = f"{output_path}.bak.{int(time.time())}"
            try:
                os.rename(output_path, backup_path)
                print(f"Created backup of problematic file at {backup_path}")
            except Exception as be:
                print(f"Failed to create backup: {be}")
    
    # 파일이 없으면 빈 리스트 반환
    return []

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
    existing_results = load_existing_ccqa_results(output_path)

    # 이미 처리된 항목을 결과로 사용 (존재할 경우)
    if existing_results and len(existing_results) > 0:
        print(f"Using existing results with {len(existing_results)} items")
        results = existing_results

    # 필터링: 아직 generated_question_1~20 중 하나라도 없는 항목만 처리 대상
    results_to_process = []
    for idx, item in enumerate(results):
        # generated_question_1부터 generated_question_20까지 하나라도 없으면 처리 대상
        if not any(f"generated_question_{i}" in item for i in range(1, 21)):
            results_to_process.append((idx, item))

    processed_count = len(results) - len(results_to_process)
    total_questions = len(results)

    print(f"Found {processed_count} already processed questions out of {total_questions}")

    if not results_to_process:
        print(f"All questions already processed. Nothing to do.")
        return results

    print(f"Will process {len(results_to_process)} remaining questions")

    # 진행 상황 표시를 위한 tqdm 객체
    with tqdm(total=len(results_to_process), desc="Applying CCQA") as pbar:
        for idx_pair in results_to_process:
            global_idx, item = idx_pair
            ccqa_item = item.copy()

            # Add marker for tracking model
            ccqa_item["used_t5_model"] = True

            # Original body (question 필드 사용)
            original_body = ccqa_item.get("question", "")

            # 각 응답에 대해 질문 생성
            for resp_idx in range(1, 21):
                response_key = f"response_{resp_idx}"
                
                if response_key not in ccqa_item:
                    continue
                    
                answer = ccqa_item.get(response_key, "").strip()
                
                if not answer:
                    continue

                try:
                    # 질문 생성
                    generated_question = generate_question_from_answer(
                        model=t5_model,
                        body=original_body,
                        answer=answer,
                        max_length=max_length
                    )
                    
                    # 생성된 질문 저장
                    ccqa_item[f"generated_question_{resp_idx}"] = generated_question
                    
                except Exception as e:
                    ccqa_item[f"generated_question_{resp_idx}"] = f"Error: {str(e)}"

            # 처리된 항목을 결과에 반영
            results[global_idx] = ccqa_item

            # 중간 진행 상황 저장
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as save_err:
                print(f"Error saving intermediate results: {save_err}")
                
            pbar.update(1)
            
            # 진행률 출력 (10개 단위 또는 처음/마지막)
            current_processed = processed_count + pbar.n
            if current_processed % 10 == 0 or current_processed == 1 or current_processed == total_questions:
                print(f"Progress: {current_processed}/{total_questions} questions completed ({(current_processed / total_questions * 100):.1f}%)")

    print(f"총 문제 수: {len(results)}")
    print(f"처리된 문제 수: {processed_count + len(results_to_process)}")

    return results


def run_ccqa_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned_2/checkpoint-11840",
    max_length: int = 50
) -> Optional[Dict]:

    try:
        start_time = time.time()
        
        # Find this model's results file
        results_filename = f"calc_mawps_{model_name}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        if not os.path.exists(results_path):
            print(f"Could not find results file for model {model_name}: {results_path}")
            return None
        
        # Load results
        results_data = load_mawps_results(results_path)
            
        if not results_data:
            print(f"No results found in {results_path}")
            return None
        
        print(f"Loaded {len(results_data)} results for model {model_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to save CCQA results
        ccqa_filename = f"calc_mawps_{model_name}.json"
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
        
        # MAWPS 결과는 리스트로 저장하고 time_info를 별도 파일로 저장
        with open(ccqa_path, 'w', encoding='utf-8') as f:
            json.dump(ccqa_results, f, ensure_ascii=False, indent=2)
            
        # 시간 정보는 별도 파일로 저장
        time_info_path = os.path.join(output_dir, f"time_info_{model_name}.json")
        with open(time_info_path, 'w', encoding='utf-8') as f:
            json.dump(time_info, f, ensure_ascii=False, indent=2)
        
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
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned_2/checkpoint-11840",
    max_length: int = 50,
    parallel: bool = True
) -> Dict[str, Dict]:

    # Find all result files in the directory
    result_files = [f for f in os.listdir(results_dir) if f.startswith("calc_mawps_") and f.endswith(".json")]
    model_names = [f.replace("calc_mawps_", "").replace(".json", "") for f in result_files]
    
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

def run_mawps_ccqa(
    results_dir: str,
    output_dir: str,
    t5_model_path: str = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned_2/checkpoint-11840",
    max_length: int = 50,
    parallel: bool = False
) -> Dict[str, Dict]:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting CCQA (using T5) for MAWPS benchmark...")
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
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/MAWPS/Result/mawps_result"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/MAWPS/Result/ccqa_result"
T5_MODEL_PATH = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned_2/checkpoint-11840"
MAX_LENGTH = 50
RUN_PARALLEL = True  # Set to True to run models in parallel

# Main execution
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_mawps_ccqa(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        t5_model_path=T5_MODEL_PATH,
        max_length=MAX_LENGTH,
        parallel=RUN_PARALLEL
    )