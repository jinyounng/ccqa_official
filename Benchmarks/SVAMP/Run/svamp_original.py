import json
import os
from typing import Dict, List, Any
from tqdm import tqdm
from datasets import load_dataset
import sys
import torch
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

# 딕셔너리 그대로 사용
MODELS = LLMRunner.AVAILABLE_MODELS

# 중복된 코드 제거
mp.set_start_method('spawn', force=True)

def format_prompt(body: str, question: str) -> str:
    """Format the problem with body and question."""
    return f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {body}\n{question} A:"""

class ModelEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # 모델 ID는 딕셔너리 값 자체입니다 (문자열)
        self.model_id = MODELS[model_name]
        
        # Load model and tokenizer
        print(f"Loading {model_name} ({self.model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
    def generate_answer(self, prompt: str) -> str:
        """Generate answer for given prompt and return response"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove the prompt from response
        
        return response

def run_benchmark(args):
    """Run the benchmark for a specific model."""
    model_name, dataset, output_dir = args
    print(f"\nRunning benchmark for {model_name}...")
    
    evaluator = ModelEvaluator(model_name)
    results = []
    
    for item in tqdm(dataset, desc=f"Processing {model_name}"):
        formatted_prompt = format_prompt(item['Body'], item['Question'])
        try:
            # Get model's response
            response = evaluator.generate_answer(formatted_prompt)
            
            result = {
                'body': item['Body'],
                'question': item['Question'],
                'equation': item['Equation'],
                'original_answer': item['Answer'],
                'generated_response': response
            }
            
        except Exception as e:
            print(f"Error processing question: {formatted_prompt}")
            print(f"Error: {str(e)}")
            result = {
                'body': item['Body'],
                'question': item['Question'],
                'equation': item['Equation'],
                'original_answer': item['Answer'],
                'generated_response': 'ERROR',
                'error': str(e)
            }
        
        results.append(result)
        
        # Save intermediate results
        output_file = os.path.join(output_dir, f"svamp_{model_name}_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Clean up
    del evaluator.model
    del evaluator.tokenizer
    torch.cuda.empty_cache()
    
    return model_name, results

def process_model_group(model_group, dataset, output_dir):
    """Process a group of models in parallel."""
    pool = mp.Pool(processes=len(model_group))
    try:
        args = [(model_name, dataset, output_dir) for model_name in model_group]
        results = pool.map(run_benchmark, args)
    finally:
        pool.close()
        pool.join()
    return dict(results)

def main(test_sample: bool = False):
    # Create output directory if it doesn't exist
    output_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/train_sets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the SVAMP dataset from HuggingFace
    print("Loading SVAMP dataset...")
    dataset = load_dataset("ChilleD/SVAMP")
    
    # Test mode with single sample if requested
    if test_sample:
        dataset['train'] = dataset['train'].select(range(1))
        print(f"\nTesting with single sample:")
        sample = dataset['train'][0]
        print(f"Body: {sample['Body']}")
        print(f"Question: {sample['Question']}")
        print(f"Equation: {sample['Equation']}")
        print(f"Answer: {sample['Answer']}")
        print(f"Formatted Prompt:")
        print(format_prompt(sample['Body'], sample['Question']))
    
    # MODELS는 이제 딕셔너리이므로 그 키들을 사용합니다
    model_names = list(MODELS.keys())
    model_groups = [model_names[i:i+6] for i in range(0, len(model_names), 6)]
    
    # Process each group of models
    for group in model_groups:
        print(f"\nProcessing model group: {group}")
        process_model_group(group, dataset['train'], output_dir)
    
    print("\nBenchmark complete.")

if __name__ == "__main__":
    try:
        main(test_sample=False)  # Set to True for testing with single sample
    finally:
        # Cleanup resources
        if mp.current_process().name == 'MainProcess':
            mp.active_children()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()