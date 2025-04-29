import os
import json
import re
import sys
import glob
import time
import csv
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Add project root path to Python path
sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

# Different prompts for experiment
PROMPT_TEMPLATES = {
    "accuracy": """Read the original text and the question carefully, and select the question, from 1 to 5, that would yield the same answer as the original question when solved. Focus on the accuracy of the answer and choose the most appropriate question number.
    Original Text: {original} 
    Options: {options}
    Please respond with ONLY the number of the question (1-5).""",
    
    "similarity": """Among the questions listed below, choose the one that is most similar to the original question. Consider the similarity in the phrasing and the information requested to make your selection.
    Original Text: {original} 
    Options: {options}
    Please respond with ONLY the number of the question (1-5).""",
    
    "intent": """Review the following questions and select the one that most closely matches the intent of the original question. Take into account the goal of the question and the type of answer it seeks to determine the most appropriate number.
    Original Text: {original} 
    Options: {options}
    Please respond with ONLY the number of the question (1-5)."""
}

# Models to use for experiments
MODELS = ["llama-3b", "qwen-3b", "llama-8b"]  # Using llama-3b twice for testing. Please replace with llama-8b when available

# Temperature values to test
TEMPERATURES = [0.3, 0.5, 0.7]

def extract_answer_with_regex(response: str) -> Optional[str]:
    """
    Extract answer using regex patterns
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted answer or None if no answer found
    """
    response_lower = response.lower()
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
        # r'the (?:correct )?answer is \\\( \\\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?) \\\)',
        # r'the (?:correct )?answer is\s*:\s*[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
        # r'the (?:correct )?answer is\s*:[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
    ]   

    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            return match.group(1).strip()
    
    # Try to find the last number in the text
    number_pattern = r'([+-]?\d+\.?\d*)\s*$'
    for line in reversed(response.split('\n')):
        match = re.search(number_pattern, line)
        if match:
            return match.group(1).strip()
    
    return None

def find_most_similar_question(original: str, questions: List[str], llm_runner, prompt_type: str, temperature: float) -> Tuple[int, float]:
    """
    Find the most similar question using LLM
    
    Args:
        original: Original question text
        questions: List of generated questions
        llm_runner: LLM runner instance
        prompt_type: Type of prompt to use (accuracy/similarity/intent)
        
    Returns:
        Tuple of (best question index, confidence score)
    """
    valid_questions = [(i, q) for i, q in enumerate(questions) if q]
    if not valid_questions:
        return -1, 0.0
    
    valid_indices = [i for i, _ in valid_questions]
    valid_questions_text = [q for _, q in valid_questions]
    questions_with_indices = "\n".join([f"{i+1}. {q}" for i, q in enumerate(valid_questions_text)])
    
    # Select prompt template based on prompt_type
    if prompt_type not in PROMPT_TEMPLATES:
        print(f"Warning: Unknown prompt type '{prompt_type}'. Using 'accuracy' instead.")
        prompt_type = "accuracy"
    
    prompt_template = PROMPT_TEMPLATES[prompt_type]
    prompt = prompt_template.format(original=original, options=questions_with_indices)
    
    try:
        responses = llm_runner.generate_responses(
            prompt=prompt, 
            num_responses=1, 
            max_new_tokens=10,
            temperature=temperature, 
            top_p=0.9, 
            parallel=False
        )
        
        response = responses[0] if responses else ""
        choice_match = re.search(r'(\d+)', response)
        
        if choice_match:
            choice = int(choice_match.group(1)) - 1
            if 0 <= choice < len(valid_indices):
                return valid_indices[choice], 1.0
        
        # If no valid choice found, return the first valid question
        return valid_indices[0], 0.5
    except Exception as e:
        print(f"Error in finding similar question: {e}")
        return valid_indices[0] if valid_indices else -1, 0.0

def get_experiment_name(model_name: str, prompt_type: str, temperature: float) -> str:
    """
    Generate a consistent experiment name
    
    Args:
        model_name: Name of the model
        prompt_type: Type of prompt used
        temperature: Temperature value used
        
    Returns:
        Experiment identifier string
    """
    return f"{model_name}_{prompt_type}_temp{temperature}"

def compare_answers(extracted_answer: str, original_answer: str) -> bool:
    """
    Compare extracted answer with original answer
    
    Args:
        extracted_answer: Answer extracted from LLM response
        original_answer: Original correct answer
        
    Returns:
        True if answers match, False otherwise
    """
    if not extracted_answer or not original_answer:
        return False
    
    # Clean and normalize both answers
    try:
        # Try to convert both to floats for numerical comparison
        extracted_float = float(extracted_answer.replace(',', '').strip())
        original_float = float(original_answer.replace(',', '').strip())
        
        # Compare with small tolerance for floating point differences
        return abs(extracted_float - original_float) < 0.01
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        extracted_clean = extracted_answer.strip().lower()
        original_clean = original_answer.strip().lower()
        return extracted_clean == original_clean

def process_ccqa_file(
    ccqa_file: str, 
    llm_model_name: str, 
    prompt_type: str,
    temperature: float,
    output_dir: str
) -> Optional[Dict]:
    """
    Process a single CCQA file with specified model, prompt and temperature
    
    Args:
        ccqa_file: Path to CCQA file
        llm_model_name: Model name to use
        prompt_type: Type of prompt to use
        temperature: Temperature value for generation
        output_dir: Directory to save results
        
    Returns:
        Experiment info dictionary or None if failed
    """
    try:
        # Create experiment identifier
        experiment_id = get_experiment_name(llm_model_name, prompt_type, temperature)
        
        # Load CCQA data
        with open(ccqa_file, 'r', encoding='utf-8') as f:
            ccqa_data = json.load(f)
        
        # Get file details
        file_name = os.path.basename(ccqa_file)
        original_model_name = file_name.split("_ccqa")[0].replace("svamp_", "")
        
        # Create a new output file name
        output_file = os.path.join(
            output_dir,
            f"{original_model_name}_ccqa_{experiment_id}.json"
        )
        
        print(f"Processing {original_model_name} model's CCQA results with {llm_model_name} and {prompt_type} prompt...")
        
        # Extract results depending on data structure
        results = ccqa_data["results"] if "results" in ccqa_data else ccqa_data
        total_items = len(results)
        
        # Initialize LLM runner
        llm_runner = LLMRunner(llm_model_name)
        
        # Initialize stats
        extraction_success = 0
        correct_answers = 0
        start_time = time.time()
        
        # Copy original data to avoid modifying it
        output_data = {"results": [item.copy() for item in results]}
        
        # Process each item
        for item_idx, item in enumerate(output_data["results"]):
            original_question = item.get("body", "") + " " + item.get("question", "")
            generated_questions = [item.get(f"generated_question_{i}", "") for i in range(1, 6)]
            
            if not all(q == "" for q in generated_questions):
                best_idx, best_score = find_most_similar_question(
                    original_question, 
                    generated_questions, 
                    llm_runner,
                    prompt_type,
                    temperature
                )
                
                if best_idx >= 0:
                    best_question_idx = best_idx + 1
                    best_response = item.get(f"response_{best_question_idx}", "")
                    ccqa_answer = extract_answer_with_regex(best_response)
                    
                    field_prefix = f"ccqa_{experiment_id}"
                    item[f"{field_prefix}_best_question_idx"] = best_question_idx
                    item[f"{field_prefix}_best_score"] = best_score
                    item[f"{field_prefix}_answer"] = ccqa_answer
                    
                    if ccqa_answer:
                        extraction_success += 1
                        
                        # Check accuracy against original answer
                        original_answer = item.get("original_answer", "")
                        is_correct = compare_answers(ccqa_answer, original_answer)
                        item[f"{field_prefix}_is_correct"] = is_correct
                        if is_correct:
                            correct_answers += 1
                else:
                    field_prefix = f"ccqa_{experiment_id}"
                    item[f"{field_prefix}_best_question_idx"] = None
                    item[f"{field_prefix}_best_score"] = 0.0
                    item[f"{field_prefix}_answer"] = None
            else:
                field_prefix = f"ccqa_{experiment_id}"
                item[f"{field_prefix}_best_question_idx"] = None
                item[f"{field_prefix}_best_score"] = 0.0
                item[f"{field_prefix}_answer"] = None
            
            # Save intermediate results
            if (item_idx + 1) % 10 == 0 or item_idx == total_items - 1:
                output_data["extraction_stats"] = {
                    "total_items": total_items,
                    "extraction_success": extraction_success,
                    "extraction_success_rate": extraction_success / total_items if total_items > 0 else 0,
                    "correct_answers": correct_answers,
                    "accuracy": correct_answers / total_items if total_items > 0 else 0,
                    "completed_items": item_idx + 1,
                    "experiment_info": {
                        "model": llm_model_name,
                        "prompt_type": prompt_type,
                        "temperature": temperature,
                        "original_model": original_model_name
                    }
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"Progress: {item_idx+1}/{total_items} ({(item_idx+1)/total_items*100:.1f}%)")
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time to stats
        output_data["extraction_stats"]["processing_time_seconds"] = processing_time
        output_data["extraction_stats"]["processing_time_minutes"] = processing_time / 60
        
        # Save final results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Experiment {experiment_id} on {original_model_name} completed:")
        print(f"- Success rate: {extraction_success}/{total_items} ({extraction_success/total_items*100:.1f}%)")
        print(f"- Processing time: {processing_time/60:.2f} minutes")
        
        return {
            "original_model": original_model_name,
            "experiment_model": llm_model_name,
            "prompt_type": prompt_type,
            "temperature": temperature,
            "output_file": output_file,
            "extraction_success": extraction_success,
            "total_items": total_items,
            "success_rate": extraction_success/total_items if total_items > 0 else 0,
            "correct_answers": correct_answers,
            "accuracy": correct_answers/total_items if total_items > 0 else 0,
            "processing_time": processing_time
        }
    
    except Exception as e:
        print(f"Error processing file {ccqa_file} with {llm_model_name} and {prompt_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results_to_csv(experiments: List[Dict], output_file: str):
    """
    Save experiment results to CSV file
    
    Args:
        experiments: List of experiment result dictionaries
        output_file: Path to save CSV file
    """
    if not experiments:
        print("No experiments to save")
        return
    
    # Define CSV fields
    fieldnames = [
        "original_model",
        "experiment_model", 
        "prompt_type",
        "temperature",
        "total_items",
        "extraction_success",
        "success_rate",
        "correct_answers",
        "accuracy",
        "processing_time_minutes"
    ]
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for exp in experiments:
            # Convert processing time to minutes
            processing_time_minutes = exp["processing_time"] / 60 if "processing_time" in exp else 0
            
            writer.writerow({
                "original_model": exp.get("original_model", ""),
                "experiment_model": exp.get("experiment_model", ""),
                "prompt_type": exp.get("prompt_type", ""),
                "temperature": exp.get("temperature", ""),
                "total_items": exp.get("total_items", 0),
                "extraction_success": exp.get("extraction_success", 0),
                "success_rate": f"{exp.get('success_rate', 0) * 100:.2f}%",
                "correct_answers": exp.get("correct_answers", 0),
                "accuracy": f"{exp.get('accuracy', 0) * 100:.2f}%",
                "processing_time_minutes": f"{processing_time_minutes:.2f}"
            })
    
    print(f"Results saved to CSV: {output_file}")

def run_all_experiments(ccqa_dir: str, output_dir: str):
    """
    Run all combinations of models and prompts on all CCQA files
    
    Args:
        ccqa_dir: Directory containing CCQA result files
        output_dir: Directory to save experiment results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CCQA files
    ccqa_files = glob.glob(os.path.join(ccqa_dir, "svamp_*_ccqa*.json"))
    if not ccqa_files:
        print(f"No CCQA files found in {ccqa_dir}")
        return
    
    print(f"Found {len(ccqa_files)} CCQA files to process")
    
    # List to store all experiment results
    all_experiments = []
    
    # For each CCQA file
    for ccqa_file in ccqa_files:
        file_name = os.path.basename(ccqa_file)
        original_model_name = file_name.split("_ccqa")[0].replace("svamp_", "")
        print(f"\nProcessing CCQA file for model: {original_model_name}")
        
        # Run all model, prompt, and temperature combinations
        for model_name in MODELS:
            for prompt_type in PROMPT_TEMPLATES.keys():
                for temperature in TEMPERATURES:
                    print(f"\nStarting experiment with {model_name} model, {prompt_type} prompt, and temperature {temperature}...")
                    
                    experiment_result = process_ccqa_file(
                        ccqa_file=ccqa_file,
                        llm_model_name=model_name,
                        prompt_type=prompt_type,
                        temperature=temperature,
                        output_dir=output_dir
                    )
                
                if experiment_result:
                    all_experiments.append(experiment_result)
    
    # Save summary of all experiments
    summary_path = os.path.join(output_dir, "experiments_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(all_experiments),
            "experiments": all_experiments
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll experiments completed. Summary saved to {summary_path}")
    
    # Save results to CSV
    csv_output_path = os.path.join(output_dir, "experiments_results.csv")
    save_results_to_csv(all_experiments, csv_output_path)
    
    # Print summary table
    print("\nExperiment Summary:")
    print("=" * 110)
    print(f"{'Original Model':<15} {'Exp. Model':<10} {'Prompt Type':<10} {'Temp':<5} {'Success Rate':<12} {'Accuracy':<12} {'Time (min)':<10}")
    print("-" * 110)
    
    for exp in all_experiments:
        print(f"{exp['original_model']:<15} {exp['experiment_model']:<10} {exp['prompt_type']:<10} "
              f"{exp['temperature']:<5.1f} {exp['success_rate']*100:>8.1f}% {exp['accuracy']*100:>8.1f}% {exp['processing_time']/60:>9.2f}")
    print("=" * 110)

if __name__ == "__main__":
    # Set input and output directories
    CCQA_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result" 
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Experiments/ccqa_t5_result"
    
    # Run all experiments
    run_all_experiments(CCQA_DIR, OUTPUT_DIR)