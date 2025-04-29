import json
import optuna
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
import os
from torch.utils.data import Subset
import time
from datetime import datetime

# 재현성을 위한 시드 설정
set_seed(42)

# 경로 설정
DATA_PATH = "/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_complete.json"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-optimized"
MODEL_NAME = "google/flan-t5-base"

# GPU 설정 확인
NUM_GPUS = torch.cuda.device_count()
log_message = f"감지된 GPU 수: {NUM_GPUS}"
print(log_message)

# 로깅 설정
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f"{OUTPUT_DIR}/optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    """로그 메시지를 파일과 콘솔에 출력"""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

log_message(f"Starting hyperparameter optimization for {MODEL_NAME} with {NUM_GPUS} GPUs")

# 1. 데이터 로드
log_message("Loading data...")
with open(DATA_PATH, "r") as f:
    raw_data = json.load(f)

# 데이터 구조 변환
processed_data = []
for item in raw_data:
    if all(k in item for k in ['dataset_type', 'question', 'response']):
        processed_item = {
            'input': item['response'],
            'output': item['question'],
            'dataset': item['dataset_type']
        }
        processed_data.append(processed_item)

log_message(f"Processed data: {len(processed_data)}/{len(raw_data)} ({len(processed_data)/len(raw_data)*100:.2f}%)")

# 데이터 검증
valid_data = []
for item in processed_data:
    if isinstance(item.get('input'), str) and isinstance(item.get('output'), str):
        if len(item['input'].strip()) > 0 and len(item['output'].strip()) > 0:
            valid_data.append(item)
log_message(f"Valid data: {len(valid_data)}/{len(processed_data)} ({len(valid_data)/len(processed_data)*100:.2f}%)")

# 데이터를 학습용과 평가용으로 분할 (90% 학습, 10% 평가)
train_data, eval_data = train_test_split(valid_data, test_size=0.1, random_state=42)
log_message(f"Train data: {len(train_data)}, Eval data: {len(eval_data)}")

# 데이터셋 객체 생성
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# 데이터 전처리 함수
def preprocess_function(tokenizer, example, max_input_length, max_target_length):
    # 데이터셋에 따라 다른 프롬프트 형식 적용
    dataset_type = example.get("dataset", "").strip()
    
    if dataset_type == "gsm8k" or dataset_type == "svamp" or dataset_type == "mawps":
        prompt = f"CRITICAL: Do not change ANY numeric values in the answer. Every number must be preserved EXACTLY in your question. Generate a question that would have this as its answer: {example['input']}"
    elif dataset_type == "CommonSenseQA":
        prompt = f"CRITICAL: From the commonsense reasoning answer provided below, recreate the original commonsense reasoning question. No need to include choices. Answer: {example['input']}"
    elif dataset_type == "StrategyQA":
        prompt = f"CRITICAL: Create a yes/no question where the answer would be '{example['input']}'."
    else:
        prompt = f"Generate the original detailed question based on the given answer: {example['input']}"
    
    target = example["output"]
    
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,  
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    targets = tokenizer(
        target,
        max_length=max_target_length,
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

# 샘플링 기능 (큰 데이터셋에서 최적화를 위한 서브셋 추출)
def sample_dataset(dataset, sample_size, seed=42):
    np.random.seed(seed)
    if len(dataset) <= sample_size:
        return dataset
    
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    return Subset(dataset, indices)

# Optuna objective 함수
def objective(trial):
    # 시작 시간 기록
    start_time = time.time()
    
    # 하이퍼파라미터 샘플링
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    per_device_batch_size = trial.suggest_int("per_device_batch_size", 8, 32, step=8)
    grad_accum_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4, step=1)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    max_input_length = trial.suggest_categorical("max_input_length", [256, 384, 512])
    max_target_length = trial.suggest_categorical("max_target_length", [64, 128, 192])
    
    # 트라이얼 정보 로깅
    log_message(f"\nStarting trial {trial.number} with parameters:")
    for key, value in trial.params.items():
        log_message(f"  {key}: {value}")
    
    # 최적화를 위한 데이터 샘플링 (시간 절약을 위해)
    # A6000 GPU 3장을 고려하여 샘플 크기 증가
    train_sample_size = min(15000, len(train_dataset))
    eval_sample_size = min(3000, len(eval_dataset))
    
    train_subset = sample_dataset(train_dataset, train_sample_size)
    eval_subset = sample_dataset(eval_dataset, eval_sample_size)
    
    log_message(f"Using {train_sample_size} training samples and {eval_sample_size} evaluation samples for optimization")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 데이터 전처리
    def preprocess(example):
        return preprocess_function(tokenizer, example, max_input_length, max_target_length)
    
    tokenized_train = train_subset.map(
        preprocess,
        remove_columns=["input", "output", "dataset"] if "dataset" in train_dataset.features else ["input", "output"],
        load_from_cache_file=False,
        num_proc=8  # 병렬 처리 증가
    )
    
    tokenized_eval = eval_subset.map(
        preprocess,
        remove_columns=["input", "output", "dataset"] if "dataset" in eval_dataset.features else ["input", "output"],
        load_from_cache_file=False,
        num_proc=8  # 병렬 처리 증가
    )
    
    # 모델 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # 훈련 아규먼트 - 멀티 GPU 설정 추가
    trial_output_dir = f"{OUTPUT_DIR}/trial-{trial.number}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=trial_output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=1,  # 최적화를 위해 1 에폭만 실행
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=warmup_ratio,
        fp16=True,  # A6000은 FP16을 지원
        report_to="none",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        # 멀티 GPU 설정
        local_rank=-1,  # 분산 학습을 위한 로컬 랭크
        dataloader_num_workers=4,  # 데이터 로더 워커 수 증가
        ddp_find_unused_parameters=False,  # DDP 성능 향상
        gradient_checkpointing=True,  # 메모리 효율성
    )
    
    # 트레이너 설정
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        ),
    )
    
    # 훈련 실행
    train_result = trainer.train()
    
    # 평가 실행
    metrics = trainer.evaluate()
    
    # 시간 측정
    elapsed_time = time.time() - start_time
    log_message(f"Trial {trial.number} completed in {elapsed_time/60:.2f} minutes")
    log_message(f"Trial {trial.number} results: eval_loss={metrics['eval_loss']:.4f}")
    
    # 메모리 정리
    del model, trainer, tokenized_train, tokenized_eval
    torch.cuda.empty_cache()
    
    return metrics["eval_loss"]

# Optuna 스터디 생성 및 실행
log_message("Creating Optuna study...")
study = optuna.create_study(direction="minimize")

# 저장된 최적화 결과가 있는지 확인
study_db_path = f"{OUTPUT_DIR}/optuna_study.db"
if os.path.exists(study_db_path):
    log_message(f"Loading previous study from {study_db_path}")
    study = optuna.load_study(study_name="t5_optimization", storage=f"sqlite:///{study_db_path}")

# 최적화 실행
log_message("Starting optimization...")
n_trials = 15  # A6000 3장 사용하므로 더 많은 trial 시도
try:
    study.optimize(objective, n_trials=n_trials)
except Exception as e:
    log_message(f"Optimization interrupted: {e}")

# 최적 하이퍼파라미터 출력
log_message("\nBest trial:")
trial = study.best_trial
log_message(f"  Value (eval_loss): {trial.value}")
log_message("  Params: ")
for key, value in trial.params.items():
    log_message(f"    {key}: {value}")

# 최적 하이퍼파라미터 저장
with open(f"{OUTPUT_DIR}/best_hyperparameters.json", "w") as f:
    json.dump(trial.params, f, indent=2)
log_message(f"Best hyperparameters saved to {OUTPUT_DIR}/best_hyperparameters.json")

# 최종 모델 학습 준비
log_message("\n준비: 최종 모델 학습...")

# 최적 파라미터 로드
with open(f"{OUTPUT_DIR}/best_hyperparameters.json", "r") as f:
    best_params = json.load(f)

# 로드된 파라미터 출력
log_message("최적 하이퍼파라미터:")
for key, value in best_params.items():
    log_message(f"  {key}: {value}")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 전체 데이터셋 전처리 (최적 파라미터 사용)
log_message("전체 데이터셋 전처리 중...")

def preprocess_final(example):
    return preprocess_function(
        tokenizer, 
        example, 
        best_params["max_input_length"], 
        best_params["max_target_length"]
    )

# 멀티 프로세싱 증가 (A6000 시스템에 맞게)
tokenized_train = train_dataset.map(
    preprocess_final,
    remove_columns=["input", "output", "dataset"] if "dataset" in train_dataset.features else ["input", "output"],
    load_from_cache_file=False,
    num_proc=12  # 병렬 처리 수 증가 (A6000 시스템 고려)
)

tokenized_eval = eval_dataset.map(
    preprocess_final,
    remove_columns=["input", "output", "dataset"] if "dataset" in eval_dataset.features else ["input", "output"],
    load_from_cache_file=False,
    num_proc=12
)

log_message(f"전체 학습 데이터셋 크기: {len(tokenized_train)}")
log_message(f"전체 평가 데이터셋 크기: {len(tokenized_eval)}")

# 모델 로드
log_message("모델 로드 중...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 최종 학습 설정
final_output_dir = f"{OUTPUT_DIR}/final_model"
os.makedirs(final_output_dir, exist_ok=True)

# A6000 3장을 활용한 분산 학습 설정
# 최종 훈련 아규먼트
final_training_args = Seq2SeqTrainingArguments(
    output_dir=final_output_dir,
    per_device_train_batch_size=best_params["per_device_batch_size"],
    per_device_eval_batch_size=best_params["per_device_batch_size"],
    gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    num_train_epochs=5,  # 최종 학습은 5 에폭으로 증가 (A6000 3장 활용)
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=best_params["warmup_ratio"],
    fp16=True,  # A6000은 FP16 지원
    report_to="none",
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=best_params["max_target_length"],
    # 멀티 GPU 설정
    local_rank=-1,  # 분산 학습을 위한 로컬 랭크
    dataloader_num_workers=4,  # 데이터 로더 워커 수 증가
    ddp_find_unused_parameters=False,  # DDP 성능 향상
    gradient_checkpointing=True,  # 메모리 효율성
    # 추가 분산 학습 설정
    deepspeed=None,  # DeepSpeed 사용 시 config 파일 경로 지정 (선택 사항)
    # sharded_ddp="zero_dp_2",  # ZeRO-2 최적화 (메모리 효율)
)

# 최종 트레이너 설정
log_message("최종 트레이너 설정 중...")
final_trainer = Seq2SeqTrainer(
    model=model,
    args=final_training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    ),
)

# 최종 학습 시작
log_message("최종 학습 시작! (A6000 3장 활용)")
try:
    # 분산 학습 시작
    final_trainer.train()
    log_message("학습 완료!")
except Exception as e:
    log_message(f"학습 실패: {e}")
    # 체크포인트 저장
    checkpoint_dir = f"{OUTPUT_DIR}/checkpoint_before_crash"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    log_message(f"충돌 전 체크포인트 저장됨: {checkpoint_dir}")
    raise

# 최종 모델 및 토크나이저 저장
final_model_dir = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
log_message(f"최종 모델 저장 완료: {final_model_dir}")

# 최적화 과정 요약
log_message("\n최적화 과정 요약:")
log_message(f"총 시도 횟수: {len(study.trials)}")
log_message(f"최적 하이퍼파라미터: {best_params}")
log_message(f"최적 평가 손실: {trial.value:.4f}")
log_message(f"최종 모델 저장 위치: {final_model_dir}")
log_message(f"A6000 GPU {NUM_GPUS}장을 활용한 최적화 프로세스 완료!")