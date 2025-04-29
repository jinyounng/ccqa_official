import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

#  1. 데이터 로드
with open("/home/jykim/Projects/CCQA_official/finetuning/combined_qa_dataset.json", "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

#  2. 토크나이저 & 모델 로드
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

#  3. PEFT (LoRA) 구성
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#  4. 전처리 함수
def preprocess(example):
    prompt = f"generate question: {example['input']}\n###\n"
    target = example["output"]
    full_text = prompt + target

    tokens = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(preprocess, remove_columns=["input", "output"])

#  5. 학습 설정
training_args = TrainingArguments(
    output_dir="./smollm2-qgen-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-4,
    save_strategy="epoch",
    logging_steps=10,
    logging_dir="./logs",
    fp16=True,
    evaluation_strategy="no"
)

#  6. Trainer 구성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 7. 저장
model.save_pretrained("./smollm2-qgen")
tokenizer.save_pretrained("./smollm2-qgen")
