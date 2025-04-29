import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# 1. 데이터 로드
with open("/home/jykim/Projects/CCQA_official/finetuning/extractor_dataset.json", "r") as f:
    raw_data = json.load(f)

# 2. 데이터 분할 (학습/검증/테스트 세트)
train_data, temp_data = train_test_split(raw_data, test_size=0.2, random_state=42)
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}, Test samples: {len(test_data)}")

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)
test_dataset = Dataset.from_list(test_data)

# 3. 모델 및 토크나이저 불러오기 - FLAN-T5 Base로 변경
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 4. 전처리 함수 개선
def preprocess(example):
    dataset_type = example.get('dataset', '')
    
    if dataset_type == 'CommonSenseQA':
        prompt = (
            "Task: Extract ONLY the FIRST correct answer from the generated response. "
            "If multiple answers appear, return ONLY the FIRST one.\n\n"
            f"Question: {example['question']}\n\n"
            f"Generated response: {example['generated_answer']}\n\n"
            "First correct answer:"
        )
        target = example["answer"]
    else:
        prompt = (
            "Task: Extract ONLY the FIRST numerical answer from this solution. "
            "If multiple numerical answers appear, extract ONLY the FIRST one that appears.\n\n"
            f"Question: {example['question']}\n\n"
            f"Solution: {example['generated_answer']}\n\n"
            "First numerical answer:"
        )
        target = example["answer"]
    
    # 입력 토큰화 (배치 처리가 아닌 개별 예제로 진행)
    input_encodings = tokenizer(
        prompt,
        max_length=768,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            target,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    input_encodings["labels"] = target_encodings["input_ids"]
    
    # tensor -> list 변환 (Trainer에서 자동 변환 가능하므로 제거해도 무방)
    for key in input_encodings:
        input_encodings[key] = input_encodings[key].squeeze().tolist()
    
    return input_encodings

# 5. 데이터셋 토큰화
print("토큰화 중...")
tokenized_train_dataset = train_dataset.map(
    preprocess, 
    remove_columns=["dataset", "question", "answer", "generated_answer"],
    batched=False,
    num_proc=4
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess, 
    remove_columns=["dataset", "question", "answer", "generated_answer"],
    batched=False,
    num_proc=4
)

tokenized_test_dataset = test_dataset.map(
    preprocess, 
    remove_columns=["dataset", "question", "answer", "generated_answer"],
    batched=False,
    num_proc=4
)

# 6. 데이터 콜레이터 생성 (패딩 처리)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # True 사용
    label_pad_token_id=-100
)

# 7. 학습 설정 - 학습률을 낮추고 fp16 비활성화
output_dir = "/home/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-answer-extractor"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-5,          # 학습률을 1e-5로 낮춤
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,                  # fp16 비활성화
    report_to="tensorboard",
    push_to_hub=False,
    remove_unused_columns=True,
    label_smoothing_factor=0.05,
    dataloader_num_workers=4,
    seed=42,
    ddp_find_unused_parameters=False
)

# 8. Trainer 구성 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 9. 학습 시작
print("학습 시작...")
trainer.train()

# 10. 테스트 세트 평가
print("테스트 세트 평가...")
test_results = trainer.evaluate(tokenized_test_dataset)
print(f"테스트 결과: {test_results}")

# 11. 모델 및 토크나이저 저장
print("모델 저장 중...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("파인튜닝 완료!")

# 12. 개선된 추론 함수
def generate_answer(question, generated_response, dataset_type=""):
    if dataset_type == "CommonSenseQA":
        prompt = (
            "Task: Extract ONLY the FIRST alphabet answer from the generated response. "
            "If multiple answers appear, return ONLY the FIRST one.\n\n"
            f"Question: {question}\n\n"
            f"Generated response: {generated_response}\n\n"
            "First answer:"
        )
    else:
        prompt = (
            "Task: Extract ONLY the FIRST numerical answer from this solution. "
            "If multiple numerical answers appear, extract ONLY the FIRST one that appears.\n\n"
            f"Question: {question}\n\n"
            f"Solution: {generated_response}\n\n"
            "First numerical answer:"
        )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    outputs = model.generate(
        input_ids, 
        max_length=64,
        min_length=1,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.2
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# 테스트 예제
test_question = (
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
    "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
    "How much more money does Betty need to buy the wallet?"
)
test_generated_answer = (
    "Betty has half the money she needs, so she has $100 / 2 = $50. "
    "Her parents give her $15, so she has $50 + $15 = $65. "
    "Her grandparents give her twice as much as her parents, so they give her 2 * $15 = $30. "
    "In total, Betty has $65 + $30 = $95. Therefore, she needs $100 - $95 = $5 more. The answer is 5."
)

print("\n테스트 예제로 추론:")
print(f"질문: {test_question}")
print(f"생성된 답변: {test_generated_answer}")
print(f"추출된 정답: {generate_answer(test_question, test_generated_answer)}")

# (선택적) 샘플별 예측 결과 저장 함수
def save_predictions(dataset, output_file):
    predictions = []
    for example in dataset:
        question = example["question"]
        generated_answer = example["generated_answer"]
        dataset_type = example.get("dataset", "")
        
        extracted_answer = generate_answer(question, generated_answer, dataset_type)
        gold_answer = example["answer"]
        
        predictions.append({
            "question": question,
            "generated_answer": generated_answer,
            "extracted_answer": extracted_answer,
            "gold_answer": gold_answer,
            "correct": extracted_answer.strip() == gold_answer.strip()
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / len(predictions) if predictions else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return predictions

# 테스트 세트에 대한 예측 저장 (원하는 경우 주석 해제 후 실행)
# predictions = save_predictions(test_dataset, f"{output_dir}/test_predictions.json")
