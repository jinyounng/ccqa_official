import json
import torch
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import concurrent.futures
import numpy as np

# 고정 경로 설정
INPUT_FILE = "/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset.json"
OUTPUT_FILE = "/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_qwen14B.json"
CACHE_DIR = "/data3/DB/LLM/Qwen2.5"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct" 
BATCH_SIZE = 8
SAVE_INTERVAL = 100
print(f"GPU 수: {torch.cuda.device_count()}")
GPU_IDS = [0, 1, 2]  # 4개의 GPU 사용
NUM_WORKERS = len(GPU_IDS)  # GPU당 하나의 워커 사용

# 환경 변수로 캐시 디렉토리 설정
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

class GPUWorker:
    def __init__(self, worker_id, model_name=MODEL_NAME):
        self.worker_id = worker_id
        self.gpu_id = GPU_IDS[worker_id % len(GPU_IDS)]  # 각 워커에 GPU 할당
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.initialize()
        
    def initialize(self):
        """각 워커가 특정 GPU에 독립적으로 모델 로드"""
        print(f"워커 {self.worker_id}가 GPU {self.gpu_id}에 모델 로드 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            padding_side='left',
            cache_dir=CACHE_DIR
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # 특정 GPU에 모델 로드
        device = f"cuda:{self.gpu_id}"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=device,  # 특정 GPU에 직접 할당
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        
        print(f"워커 {self.worker_id}가 GPU {self.gpu_id}에 모델 로드 완료")
    
    def generate_batch(self, prompts, max_length=100, temperature=0.2, top_p=0.9):
        """배치 처리를 통해 여러 질문을 동시에 생성"""
        device = f"cuda:{self.gpu_id}"
        input_ids_list = []
        attention_mask_list = []
        prompt_lengths = []
        
        # 각 프롬프트를 토큰화하고 입력 준비
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # 입력을 GPU로 이동
            input_ids_list.append(inputs.input_ids.to(device))
            attention_mask_list.append(inputs.attention_mask.to(device))
            prompt_lengths.append(len(prompt))
        
        # 배치 처리를 위한 입력 패딩
        max_input_length = max([ids.size(1) for ids in input_ids_list])
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            padding_length = max_input_length - input_ids.size(1)
            # 패딩을 GPU에서 직접 수행
            padded_input_ids.append(torch.cat([
                torch.ones((1, padding_length), dtype=torch.long, device=device) * self.tokenizer.pad_token_id,
                input_ids
            ], dim=1))
            padded_attention_masks.append(torch.cat([
                torch.zeros((1, padding_length), dtype=torch.long, device=device),
                attention_mask
            ], dim=1))
        
        # 배치 입력 생성
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_attention_mask = torch.cat(padded_attention_masks, dim=0)
        
        # 메모리 최적화를 위해 불필요한 변수 해제
        del padded_input_ids, padded_attention_masks, input_ids_list, attention_mask_list
        
        # 배치 생성
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        # 생성된 텍스트 디코딩 및 프롬프트 부분 제거
        generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        questions = []
        
        for i, text in enumerate(generated_texts):
            # 원래 프롬프트 부분을 제거하고 생성된 질문만 추출
            question = text[prompt_lengths[i]:].strip()
            
            # 후처리
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1].strip()
            
            questions.append(question)
        
        # 메모리 정리
        del batch_input_ids, batch_attention_mask, outputs, generated_texts
        torch.cuda.empty_cache()
        
        return questions

def create_prompt(example, dataset_type):
    """데이터셋 유형에 따라 적절한 프롬프트 생성"""
    
    # 수학 문제 Few-shot 예시
    math_examples = [
        {
            "response": " Marco's dad's strawberries weighed 11 pounds. Together their strawberries weighed 30 pounds. Marco's strawberries weigh 30 - 11 = 19 pounds. The answer is 19.",
            "question": "Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?"
        },
        {
            "response": "There were 50 red macaroons and 40 green macarons. If Fran ate 15 green macaroons, she ate 15 * 2 = 30 red macaroons. Now she has 40 - 15 = 25 green macarons and 50 - 30 = 20 red macaroons. So Fran has 25 + 20 = 45 macaroons left. The answer is 45.",
            "question": "Fran baked 50 red macaroons and 40 green macarons. How many macaroons will remain if Fran ate 15 green macaroons and twice as many red macaroons as green macaroons?"
        },
        {
            "response": "Let's say Pop's spending is P. Then Crackle's spending is 3P. And since Snap spends twice as much as Crackle, Snap's spending is 2 * 3P = 6P. The total spending is P + 3P + 6P = 10P. We are given that the total spending is $150. We can set up the equation 10P = 150. Solving for P, we get P = 150 / 10 = 15. So Pop spent $15. The answer is 15.",
            "question": "Snap, Crackle, and Pop spend $150 on cereal in a grocery store.  Snap spends twice as much as Crackle.  Crackle spends 3 times as much as Pop.  How much did Pop spend?"
        }
    ]
    
    # 상식 문제 Few-shot 예시
    commonsense_examples = [
        {
            "response": " Jewelry store, B: Neck, C: Jewlery box, D: Jewelry box, E: Boutique\n\nThe answer must be a location where one can purchase jewelry. Among the given options, a jewelry store is specifically designed for this purpose. So the answer is (A).",
            "question": "To locate a choker not located in a jewelry box or boutique where would you go?"
        },
        {
            "response": " The answer should be the effect of playing soccer for a long time. Of the above choices, the best answer is E. So the answer is E.",
            "question": "What does playing soccer for a long time lead to?"
        },
        {
            "response": " Mixed martial arts is not totally original from Roman Colosseum games. While it originated in ancient Rome, it evolved over time. It is a relatively modern discipline. Thus, it is not a direct copy from Roman Colosseum games. So the answer is no.",
            "question": "Is Mixed martial arts totally original from Roman Colosseum games?"
        },
        {
            "response": " The answer must be related to the grill. Of the above choices, only barbecue refers to slow cooking with a grill. So the answer is (E).",
            "question": "What is it called when you slowly cook using a grill?"
        }
    ]
    
    # StrategyQA Few-shot 예시
    strategyqa_examples = [
        {
            "response": " Mixed martial arts is not totally original from Roman Colosseum games. While it originated in ancient Rome, it evolved over time. It is a relatively modern discipline. Thus, it is not a direct copy from Roman Colosseum games. So the answer is no.",
            "question": "Is Mixed martial arts totally original from Roman Colosseum games?"
        },
        {
            "response": " Hawaiian cuisine includes dishes such as poke, laulau, and kalua pig. These dishes contain fish and pork. Thus, the cuisine of Hawaii is not suitable for a vegan. So the answer is no.",
            "question": "Is the cuisine of Hawaii suitable for a vegan?"
        },
        {
            "response": " Flying fish have excellent eyesight. In fact, they have eyes that are almost 4 times the size of those of humans. Thus, flying fish have a highly developed visual system. So the answer is yes.",
            "question": "Do flying fish have good eyesight?"
        }
    ]
    base_prompt = """
Please don't generate information that is not in the answer.

Important: 
1. Generate ONLY the question itself, without any additional instructions or meta-commentary.
2. Do not include phrases like "Please correct...", "Explain...", or any other directions. 
3. End your response immediately after the actual question.
4. The question should be in the form of a direct problem statement that a student would solve.
5. For the last, DO NOT include the answer to the question.

For example, good responses would be like:
- "How many bananas are in each group?"
- "What is the final temperature of the mixture?"
- "How long will it take for both trains to meet?"

Bad responses would include meta-instructions like:
- "How many bananas are in each group? Please show your work."
- "What is the final temperature? Explain your reasoning. To solve this, we need to..."""

    # 데이터셋 유형에 따라 다른 Few-shot 예시와 프롬프트 사용
    if dataset_type.lower() in ["gsm8k", "svamp", "mawps"]:
        # 수학 문제용 프롬프트
        examples_text = "\n\n".join([f"Answer: {ex['response']}\nQuestion: {ex['question']}" for ex in math_examples])
        prompt = f"""Generate the original question for the given math problem answer. Do not change any numeric values in the answer. \n{base_prompt}\n
{examples_text}

Answer: {example['response']}
Question:"""

    elif dataset_type.lower() in ["commonsenseqa", "commonsense_qa"]:
        # 상식 문제용 프롬프트
        examples_text = "\n\n".join([f"Answer: {ex['response']}\nQuestion: {ex['question']}" for ex in commonsense_examples])
        prompt = f"""From the commonsense reasoning answer provided below, recreate the original question. Do not include choices in your question. \n{base_prompt}\n

{examples_text}

Answer: {example['response']}
Question:"""

    elif dataset_type.lower() in ["strategyqa", "strategy_qa"]:
        # StrategyQA 데이터셋용 프롬프트 (Yes/No 질문)
        examples_text = "\n\n".join([f"Answer: {ex['response']}\nQuestion: {ex['question']}" for ex in strategyqa_examples])
        prompt = f"""Create a yes/no question that would have this as its answer.\nPlease don't generate information that is not in the answer.\n{base_prompt}\n

{examples_text}

Answer: {example['response']}
Question:"""

    else:
        # 기본 프롬프트
        prompt = f"Based on this answer, what was the original question?\nAnswer: {example['response']}\nQuestion:"
    
    return prompt

def process_batch_with_worker(worker_id, batch_items):
    """워커에게 배치 처리를 위임"""
    global gpu_workers
    
    batch_prompts = [create_prompt(item, item.get('dataset_type', 'unknown')) for item in batch_items]
    # 워커 ID에 해당하는 워커 사용
    worker = gpu_workers[worker_id]
    
    # 배치 생성
    batch_questions = worker.generate_batch(batch_prompts)
    
    # 결과 반환
    return list(zip(batch_items, batch_questions))

def fill_missing_questions(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """데이터셋에서 question이 누락된 항목을 멀티 GPU 배치 처리로 생성하여 채움"""
    global gpu_workers
    
    # 데이터셋 로드
    print(f"데이터셋 로드 중: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 질문이 누락된 항목 필터링
    items_to_process = [item for item in dataset if 'question' not in item]
    total_items = len(dataset)
    missing_items = len(items_to_process)
    print(f"전체 항목 수: {total_items}")
    print(f"질문이 누락된 항목 수: {missing_items}")
    
    # 중간 저장 파일 경로
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    
    # GPU 워커 초기화 - 각 GPU당 하나의 워커
    gpu_workers = [GPUWorker(worker_id) for worker_id in range(NUM_WORKERS)]
    print(f"{len(gpu_workers)}개의 GPU 워커 초기화 완료. 각 워커는 GPU {GPU_IDS}에 할당됨")
    
    # 배치 단위로 분할
    batches = []
    for i in range(0, len(items_to_process), BATCH_SIZE):
        batches.append(items_to_process[i:i+BATCH_SIZE])
    
    print(f"작업을 {len(batches)}개 배치로 분할 (배치 크기: {BATCH_SIZE})")
    
    # 결과 저장을 위한 리스트
    processed_count = 0
    results_mapping = {}  # 원본 항목 -> 생성된 질문 매핑
    
    # 스레드풀을 사용한 병렬 처리
    # 각 GPU에 독립적인 모델이 있으므로 워커 수만큼 스레드 사용
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 각 배치를 GPU 워커에 할당 (라운드 로빈 방식)
        future_to_batch_idx = {
            executor.submit(process_batch_with_worker, i % NUM_WORKERS, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # 완료된 작업 처리
        for future in tqdm(concurrent.futures.as_completed(future_to_batch_idx), total=len(batches), desc="배치 처리 중"):
            batch_idx = future_to_batch_idx[future]
            try:
                results = future.result()
                
                # 결과 매핑에 추가
                for item, question in results:
                    results_mapping[id(item)] = question
                
                processed_count += len(results)
                
                # 진행 상황 표시
                elapsed_time = time.time() - start_time
                items_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                estimated_time = (missing_items - processed_count) / items_per_second if items_per_second > 0 else 0
                
                print(f"처리 완료: {processed_count}/{missing_items} 항목 ({processed_count/missing_items*100:.2f}%)")
                print(f"속도: {items_per_second:.2f} 항목/초, 남은 예상 시간: {estimated_time/60:.2f} 분")
                
                # 일정 간격으로 중간 저장
                if processed_count % SAVE_INTERVAL == 0 or processed_count == missing_items:
                    # 원본 데이터셋에 결과 반영
                    for item in items_to_process:
                        if id(item) in results_mapping:
                            item_index = dataset.index(item)
                            dataset[item_index]['question'] = results_mapping[id(item)]
                            dataset[item_index]['question_generated'] = True
                    
                    print(f"{processed_count}/{missing_items} 항목 처리 후 체크포인트 저장")
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            except Exception as e:
                print(f"배치 {batch_idx} 처리 중 오류 발생: {e}")
    
    # 최종 결과 반영
    for item in items_to_process:
        if id(item) in results_mapping:
            item_index = dataset.index(item)
            dataset[item_index]['question'] = results_mapping[id(item)]
            dataset[item_index]['question_generated'] = True
    
    # 최종 결과 저장
    print(f"처리된 데이터셋 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"처리 완료. {processed_count}개 항목에 대한 질문 생성 완료.")
    print(f"총 소요 시간: {total_time/60:.2f} 분, 평균 속도: {processed_count/total_time:.2f} 항목/초")
    
    return dataset

if __name__ == "__main__":
    # 기존 체크포인트가 있는지 확인
    checkpoint_file = OUTPUT_FILE.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        print(f"체크포인트 파일 발견: {checkpoint_file}")
        user_input = input("체크포인트에서 이어서 진행하시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            INPUT_FILE = checkpoint_file
            print(f"체크포인트에서 이어서 진행: {checkpoint_file}")
    
    # GPU 워커 초기화
    gpu_workers = []
    
    try:
        fill_missing_questions()
    finally:
        # 명시적으로 CUDA 메모리 정리
        for worker in gpu_workers:
            del worker.model
            del worker.tokenizer
        
        gpu_workers = []
        torch.cuda.empty_cache()