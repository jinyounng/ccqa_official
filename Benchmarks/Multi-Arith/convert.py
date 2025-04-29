import os
import json
import glob

def transform_json(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each item in the list
        for item in data:
            # Extract responses and flatten them
            if 'all_responses' in item:
                for response_key, response_value in item['all_responses'].items():
                    item[response_key] = response_value
                del item['all_responses']
            
            # Extract questions and rename them
            if 'all_questions' in item:
                for question_key, question_value in item['all_questions'].items():
                    # Rename question_1 to generated_question_1, etc.
                    new_key = 'generated_' + question_key
                    item[new_key] = question_value
                del item['all_questions']
        
        # Write the transformed data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, f"Successfully transformed {file_path}"
    
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def process_directory(directory_path):
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        return f"No JSON files found in {directory_path}"
    
    results = []
    for json_file in json_files:
        success, message = transform_json(json_file)
        results.append(message)
    
    return "\n".join(results)

if __name__ == "__main__":
    directory_path = "/data3/jykim/Projects/CCQA_official/Benchmarks/Multi-Arith/Result/ccqa_multiArith_few_shot_result"
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
    else:
        result = process_directory(directory_path)
        print(result)