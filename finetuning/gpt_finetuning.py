import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

#  1. 데이터 로드
with open("/home/jykim/Projects/CCQA_official/finetuning/combined_qa_dataset.json", "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

#  2. 모델/토크나이저 로딩
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

#  3. LoRA 구성
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#  4. 전처리 함수 (prompt + output 붙이기)
def preprocess(example):
    prompt = f"generate question: {example['input']}\n###\n{example['output']}"
    tokens = tokenizer(prompt, max_length=256, padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(preprocess, remove_columns=["input", "output"])

#  5. 학습 설정
training_args = TrainingArguments(
    output_dir="./gptneo-qgen-peft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=5e-4,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    report_to="none"
)

#  6. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

#  7. 학습
trainer.train()

# 8. 모델 저장
model.save_pretrained("./gptneo-qgen")
tokenizer.save_pretrained("./gptneo-qgen")