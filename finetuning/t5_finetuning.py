import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 재현성을 위한 시드 설정
set_seed(42)

# 1. 데이터 로드( 직접 만든 데이터셋 사용 )
with open("C://Projects//CCQA_official//finetuning//qa_dataset_checkpoint.json", "r") as f:
    raw_data = json.load(f)

# 데이터 구조 변환 - 데이터셋의 실제 구조에 맞게 조정
# 'question'을 'output'으로, 'answer'를 'input'으로 사용
processed_data = []
for item in raw_data:
    # 필요한 필드가 있는지 확인
    if all(k in item for k in ['dataset_type', 'question', 'response']):
        # 학습을 위한 구조로 변환 (질문 생성이므로 답변->질문 방향으로 학습)
        processed_item = {
            'input': item['response'],        # 답변을 입력으로
            'output': item['question'],     # 질문을 출력으로
            'dataset': item['dataset_type']      # 데이터셋 정보 유지
        }
        processed_data.append(processed_item)

print(f"Processed data: {len(processed_data)}/{len(raw_data)} ({len(processed_data)/len(raw_data)*100:.2f}%)")

# 데이터 검증 - NaN 값이나 문제가 있는 샘플 필터링
valid_data = []
for item in processed_data:
    if isinstance(item.get('input'), str) and isinstance(item.get('output'), str):
        if len(item['input'].strip()) > 0 and len(item['output'].strip()) > 0:
            valid_data.append(item)
print(f"Valid data: {len(valid_data)}/{len(processed_data)} ({len(valid_data)/len(processed_data)*100:.2f}%)")

# 데이터를 학습용과 평가용으로 분할 (90% 학습, 10% 평가)
train_data, eval_data = train_test_split(valid_data, test_size=0.1, random_state=42)

# Dataset 객체 생성
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# 2. 모델 및 토크나이저 불러오기
model_name = "google/flan-t5-base"  # flan-t5-base 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 수학 문제용 few-shot 예시 추가
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

# 3. 전처리
def preprocess(example):
    # 데이터셋에 따라 다른 프롬프트 형식 적용
    dataset_type = example.get("dataset", "").strip()
    
    if dataset_type == "gsm8k" or dataset_type == "svamp" or dataset_type == "mawps":
        # GSM8K, SVAMP, MAWPS용 few-shot 프롬프트
        prompt = f"""CRITICAL: Do not change ANY numeric values in the answer. Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. And DO NOT include the answer to the question.

{math_fewshot_examples}

Now, generate a question for this answer: {example['input']}"""

    elif dataset_type == "CommonSenseQA":
        # CommonSenseQA 데이터셋용 프롬프트 (선택지 없이 질문만 생성)
        prompt = f"CRITICAL: From the commonsense reasoning answer provided below, recreate the original commonsense reasoning question. No need to include choices. And DO NOT include the answer to the question.\n\nGenerate a question that would have this as its answer: {example['input']}\n"

    elif dataset_type == "StrategyQA":
        # StrategyQA 데이터셋용 프롬프트 
        prompt = f"CRITICAL: Create a yes/no question. And DO NOT include the answer to the question. Generate a question that would have this as its answer: '{example['input']}'."

    else:
        # 기본 프롬프트 형식 (데이터셋이 명시되지 않은 경우)
        prompt = f"Generate the original detailed question based on the given answer: {example['input']}"
    
    target = example["output"]
    
    inputs = tokenizer(
        prompt,
        max_length=512,  # Few-shot 예제가 추가되었으므로 최대 길이 증가
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    targets = tokenizer(
        target,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # T5 모델의 라벨링
    labels = targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # 패딩 토큰을 -100으로 마스킹
    
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": labels[0]
    }

# 학습 및 평가 데이터셋 전처리
tokenized_train_dataset = train_dataset.map(
    preprocess,
    remove_columns=["input", "output", "dataset"],
    batched=False
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=["input", "output", "dataset"],
    batched=False
)

# 4. 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="C://Projects//CCQA_official//finetuning//Finetuned_models//flan-t5-base-fewshot",
    per_device_train_batch_size=16,     # 배치 크기 16으로 증가
    per_device_eval_batch_size=32,      # 평가 배치 크기도 증가
    gradient_accumulation_steps=2,      # 누적 단계 감소 (더 큰 배치 크기로 인해)
    learning_rate=5e-5,                
    num_train_epochs=5,                
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    max_grad_norm=1.0,                
    fp16=False,                         
    report_to="none",
    weight_decay=0.01,
    lr_scheduler_type="cosine",        
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=False,      # 더 큰 배치와 충분한 메모리로 비활성화 가능
)

# 5. Trainer 구성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
)

# 6. 학습 시작
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    # 모델을 저장하여 지금까지의 학습을 보존
    model.save_pretrained("C://Projects//CCQA_official//finetuning//Finetuned_models//flan-t5-base-fewshot-checkpoint")
    tokenizer.save_pretrained("C://Projects//CCQA_official//finetuning//Finetuned_models//flan-t5-base-fewshot-checkpoint")
    print("Saved checkpoint of the model before crash")
    raise

# 7. 저장
model.save_pretrained("C://Projects//CCQA_official//finetuning//Finetuned_models//flan-t5-base-fewshot-fewshot")
tokenizer.save_pretrained("C://Projects//CCQA_official//finetuning//Finetuned_models//flan-t5-base-fewshot-fewshot")