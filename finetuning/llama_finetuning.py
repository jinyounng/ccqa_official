import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import torch

print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")
# 1. 데이터 로드
with open("/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_qwen14B.json", "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

# 데이터셋 구조 확인
print("데이터셋 구조 확인:")
print(dataset[0])

# 2. 데이터셋 분할 (훈련:검증 = 9:1)
train_val_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]

# 3. 모델 및 토크나이저 불러오기
model_name = "meta-llama/Llama-3.2-1B"  # Llama-3.2-1B 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},  # 논리적 0번 GPU(6번)만 사용
    torch_dtype=torch.float16,  # 메모리 절약을 위해 반정밀도 사용
    low_cpu_mem_usage=True      # CPU 메모리 사용량 최소화
)

# Llama 토크나이저 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 오른쪽 패딩 설정

# 추가된 프롬프트 지침 (일반 데이터셋용)
prompt_instructions = """Please don't generate information that is not in the answer.
Important: 1. Generate ONLY the question itself, without any additional instructions or meta-commentary. 
2. Do not include phrases like "Please correct...", "Explain...", or any other directions. 
3. End your response immediately after the actual question. 4. The question should be in the form of a direct problem statement that a student would solve. 
5. For the last, DO NOT include the answer to the question.
For example, good responses would be like: - "How many bananas are in each group?" - "What is the final temperature of the mixture?" - "How long will it take for both trains to meet?"
Bad responses would include meta-instructions like: - "How many bananas are in each group? Please show your work." - "What is the final temperature? Explain your reasoning. To solve this, we need to..."""

# 수학 문제용 few-shot 예시
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

# 4. 전처리 함수
def preprocess(example):
    # 데이터셋 타입에 따라 프롬프트 설정
    dataset_type = example['dataset_type']
    
    if dataset_type == "gsm8k" or dataset_type == "svamp" or dataset_type == "mawps":
        # GSM8K와 SVAMP, MAWPS용 프롬프트 - 수학 문제 생성 (few-shot 방식)
        prompt = f"""CRITICAL: Do not change ANY numeric values in the answer. Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. And DO NOT include the answer to the question.

{math_fewshot_examples}

Now, generate a question for this answer: {example['response']}"""
        
    elif dataset_type == "CommonSenseQA":
        # CommonSenseQA 데이터셋용 프롬프트 (선택지 없이 질문만 생성)
        prompt = f"{prompt_instructions}\n\nCRITICAL: From the commonsense reasoning answer provided below, recreate the original commonsense reasoning question. No need to include choices. And DO NOT include the answer to the question. \n\n Generate a question that would have this as its answer: {example['response']}"
        
    elif dataset_type == "StrategyQA":
        # StrategyQA 데이터셋용 프롬프트 
        prompt = f"{prompt_instructions}\n\nCRITICAL: Create a yes/no question, And DO NOT include the answer to the question. \n\n Generate a question that would have this as its answer: '{example['response']}'."
        
    else:
        # 기본 프롬프트 형식 (데이터셋이 명시되지 않은 경우)
        prompt = f"{prompt_instructions}\n\nGenerate the original detailed question based on the given answer: {example['response']}"
    
    # 타겟 데이터는 원래 질문
    target = example['question']
    
    # 전체 텍스트 (프롬프트 + 타겟)
    full_text = f"{prompt}\n\nQuestion: {target}{tokenizer.eos_token}"
    
    # 토큰화
    inputs = tokenizer(
        full_text,
        max_length=512,  # 전체 시퀀스 길이 설정
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 라벨 설정 - 인풋과 동일하게 설정하고 프롬프트 부분은 -100으로 마스킹
    labels = inputs["input_ids"].clone()
    
    # 프롬프트 부분 토큰화 (손실 계산에서 제외할 부분)
    prompt_tokens = tokenizer(
        prompt + "\n\nQuestion:",
        add_special_tokens=False,  # 특수 토큰 추가 안함
        return_tensors="pt"
    )
    
    # 프롬프트 길이
    prompt_len = prompt_tokens["input_ids"].shape[1]
    
    # 프롬프트 부분을 -100으로 마스킹 (손실 계산에서 제외)
    labels[0, :prompt_len] = -100
    
    inputs["labels"] = labels
    
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": inputs["labels"][0]
    }

# 5. 데이터셋 전처리
print("전처리 중... 훈련 데이터셋")
tokenized_train_dataset = train_dataset.map(
    preprocess,
    remove_columns=["model", "dataset_type", "response", "is_correct", "question_generated", "question"],  # 실제 컬럼 이름으로 수정
    batched=False,
    num_proc=4  # 다중 프로세스 사용으로 전처리 속도 향상
)

print("전처리 중... 검증 데이터셋")
tokenized_val_dataset = val_dataset.map(
    preprocess,
    remove_columns=["model", "dataset_type", "response", "is_correct", "question_generated", "question"],  # 실제 컬럼 이름으로 수정
    batched=False,
    num_proc=4
)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="/home/jykim/Projects/CCQA_official/Finetuned_models/llama-1b-fullfinetuned",
    per_device_train_batch_size=16,  # 배치 크기 줄임 (메모리 효율성 향상)
    per_device_eval_batch_size=16,  # 평가 배치 크기도 줄임
    gradient_accumulation_steps=2,  # 그라디언트 누적 단계 늘림
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=20,
    logging_dir="/home/jykim/Projects/CCQA_official/Finetuned_models/logs",
    save_strategy="epoch",
    save_total_limit=2,             # 저장할 체크포인트 수 제한
    evaluation_strategy="epoch",    # 에폭마다 평가
    eval_steps=None,                # epoch마다 평가하므로 None 설정
    load_best_model_at_end=True,    # 최고 성능 모델 로드
    metric_for_best_model="eval_loss", # 검증 손실 기준 최적 모델 선택
    greater_is_better=False,        # 손실이므로 낮을수록 좋음
    fp16=False,                     # FP16 비활성화 - 그라디언트 스케일링 오류 방지
    bf16=False,                     # BF16도 비활성화 (혼란 방지)
    report_to="tensorboard",        # 텐서보드로 결과 기록
    warmup_ratio=0.1,               # 데이터셋이 크므로 비율 기반 웜업 사용
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    # 메모리 효율성을 위한 추가 설정
    gradient_checkpointing=True,    # 메모리 절약을 위한 그라디언트 체크포인팅
    optim="adamw_8bit",             # 8비트 AdamW 최적화 사용 (메모리 효율성 향상)
)

# 7. 조기 종료 콜백 설정
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,      # 2 에폭 동안 개선 없으면 중지
    early_stopping_threshold=0.0    # 손실 감소 없으면 조기 종료
)

# 8. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
    callbacks=[early_stopping_callback]
)

# 9. 학습 시작
print("학습 시작...")
trainer.train()

# 10. 최종 모델 저장
print("최종 모델 저장 중...")
model.save_pretrained("/data3/jykim/Projects/CCQA_official/Finetuned_models/llama-1b-fewshot")
tokenizer.save_pretrained("/data3/jykim/Projects/CCQA_official/Finetuned_models/llama-1b-fewshot")

print("학습 완료!")