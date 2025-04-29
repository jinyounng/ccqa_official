import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random

# 모델과 토크나이저 로드
MODEL_PATH = '/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fewshot'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

# 데이터셋 로드
DATASET_PATH = '/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_qwen14B.json'
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)

# 데이터셋 별로 분류
dataset_groups = {}
for item in dataset:
    dataset_name = item['dataset_type']
    if dataset_name not in dataset_groups:
        dataset_groups[dataset_name] = []
    dataset_groups[dataset_name].append(item)

# 테스트할 데이터셋 목록
target_datasets = ['gsm8k', 'commonsenseqa', 'svamp', 'strategyqa','mawps']
math_fewshot_examples = """
Example 1:
Answer: To find out how many more push-ups Zachary did than David, subtract David's number of push-ups from Zachary's number of push-ups: 51 - 44 = 7. The answer is 7.
Question: Zachary did 51 push-ups and David did 44 push-ups in gym class today. How many more push-ups did Zachary do than David?

Example 2:
Answer: Marco's dad's strawberries weighed 11 pounds. Together their strawberries weighed 30 pounds. That means Marco's strawberries weighed 30 - 11 = 19 pounds. The answer is 19.
Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?

Example 3:
Answer: There are 3 chapters, each with the same number of pages. The total number of pages is 594. To find the number of pages per chapter, we divide the total number of pages by 3. 594 / 3 = 198. The answer is 198.
Question: Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?

Example 4:
Answer: The total cost for the cabin rental for 2 weeks is $125 * 14 = $1750. The pet fee is $100. So the total cost is $1750 + $100 = $1850. The service/cleaning fee is 20% of $1850, which is 0.20 * $1850 = $370. So the total cost is $1850 + $370 = $2220. The security deposit is 50% of $2220, which is 0.50 * $2220 = $1110. The answer is $1110.
Question: Lana and Mike are taking their dog and renting a cabin in the mountains for 2 weeks. The daily rate is $125.00 There is a $100.00 pet fee. There is also a 20% service/cleaning fee for the rental. They need to pay 50% of the entire bill as a security deposit. How much is their security deposit?
"""
# 프롬프트 템플릿
PROMPT_TEMPLATE = (
    "CRITICAL: Do not change ANY numeric values in the answer. "
    "Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. "
    "Generate a question that would have this as its answer: "
    "answer : {}"
)



def generate_question(input_text):
    prompt = PROMPT_TEMPLATE.format(input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=False,
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# 결과 저장용
results = {}

for dataset_name in target_datasets:
    print(f"\n\n===== Testing {dataset_name} dataset =====\n")
    results[dataset_name] = []
    
    # 데이터셋 존재 여부 확인
    if dataset_name not in dataset_groups:
        print(f"Dataset {dataset_name} not found in the combined dataset!")
        continue
    
    # 3개 샘플 랜덤 추출 (3개 미만이면 전체)
    samples = dataset_groups[dataset_name]
    if len(samples) > 3:
        samples = random.sample(samples, 3)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nExample {i}:")
        input_answer = sample['response']
        ref_question = sample['question']
        
        truncated_ans = input_answer[:100] + "..." if len(input_answer) > 100 else input_answer
        print(f"Input (answer): {truncated_ans}")
        print(f"Reference question: {ref_question}")
        
        try:
            generated_question = generate_question(input_answer)
            print(f"Generated question: {generated_question}")
            
            results[dataset_name].append({
                "input": input_answer,
                "reference_question": ref_question,
                "generated_question": generated_question
            })
        except Exception as e:
            print(f"Error generating question: {e}")

with open('flan_t5_base_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nTesting completed. Results saved to flan_t5_base_test_results.json")