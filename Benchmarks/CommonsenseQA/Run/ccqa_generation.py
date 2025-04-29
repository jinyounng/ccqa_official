import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from transformers import T5Tokenizer, T5ForConditionalGeneration
import multiprocessing as mp
import math

# 디렉토리 설정
model_path = "/home/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen"
input_dir = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/commonsenseqa_result"
output_dir = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results"
os.makedirs(output_dir, exist_ok=True)

# 전역 모델 변수
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 프롬프트 템플릿 정의
PROMPT_TEMPLATE = (
    "Given this answer text, create a clear and specific question that would naturally "
    "lead to this as its answer. Focus on the main concept being explained. "
    "Answer: {}"
)

def init_model():
    """각 프로세스 시작 시 모델 로딩"""
    global tokenizer, model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)


def generate_question(text: str) -> str:
    """텍스트로부터 질문 생성"""
    global tokenizer, model
    try:
        # 프롬프트 템플릿 적용
        prompt = PROMPT_TEMPLATE.format(text.strip())
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=64,
            do_sample=False,)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ Generation Error]: {e}")
        return "[Generation Error]"


def process_file(filename: str):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_list_format = isinstance(data, list)
    items = data if is_list_format else data.get("results", [])

    # Count total responses by looking for response_X fields
    total_responses = 0
    for item in items:
        total_responses += len([k for k in item.keys() if k.startswith("response_")])
    
    pbar = tqdm(total=total_responses, desc=f"Generating {filename[:30]}", leave=False)

    for idx, item in enumerate(items):
        # Get all response fields (response_1, response_2, etc.)
        response_fields = [k for k in item.keys() if k.startswith("response_")]
        
        for field in response_fields:
            # Extract the number from the field name (e.g., "response_1" -> "1")
            response_num = field.split("_")[1]
            
            # Check if the corresponding regenerated question already exists
            if f"regenerated_question_{response_num}" in item:
                pbar.update(1)
                continue  # Skip if already generated
            
            # Generate question from the response
            question = generate_question(item[field])
            item[f"regenerated_question_{response_num}"] = question

            # Save after each update
            if is_list_format:
                data[idx] = item
            else:
                data["results"][idx] = item

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            pbar.update(1)

    pbar.close()
    return filename


def process_batch(file_batch):
    """배치 내의 파일들을 처리하는 함수"""
    for filename in file_batch:
        process_file(filename)
    return len(file_batch)


def run_parallel_generation(num_groups=1):
    """파일들을 그룹으로 나누어 병렬 처리 (각 그룹은 파일 수만큼 프로세스 사용)"""
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
    
    # 파일들을 지정된 그룹 수로 분할
    batch_size = math.ceil(len(all_files) / num_groups)
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print(f"전체 {len(all_files)}개 파일을 {num_groups}개 그룹으로 나누어 처리 ({len(file_batches)}개 배치)")
    
    for batch_idx, batch in enumerate(file_batches):
        # 배치 내 파일 수만큼 프로세스 생성 (최소 1개)
        num_processes = len(batch)
        print(f"배치 {batch_idx+1}/{len(file_batches)} 처리 중... ({len(batch)}개 파일, {num_processes}개 프로세스)")
        
        # 각 배치 내에서 병렬 처리
        with Pool(processes=num_processes, initializer=init_model) as pool:
            for _ in tqdm(pool.imap_unordered(process_file, batch), 
                          total=len(batch), 
                          desc=f"배치 {batch_idx+1} 진행"):
                pass
    
    print("전체 질문 재생성 완료!")


# 예시 실행
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # 예: 그룹 수 2로 지정 (각 그룹 내 파일 수만큼 프로세스 자동 할당)
    run_parallel_generation(num_groups=1)