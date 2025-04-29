import json
import random
import torch
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 데이터 로드
def load_data():
    with open("/data3/jykim/Projects/CCQA_official/finetuning/qa_dataset_qwen14B.json", "r") as f:
        raw_data = json.load(f)
    return raw_data

# 2. 각 벤치마크별로 분류하고 샘플링
def sample_test_data(data, samples_per_benchmark=3):
    # 벤치마크별로 데이터 분류
    benchmark_data = defaultdict(list)
    for item in data:
        benchmark = item.get("dataset_type", "unknown")
        benchmark_data[benchmark].append(item)
    
    # 각 벤치마크별로 샘플 추출
    test_samples = []
    for benchmark, items in benchmark_data.items():
        if len(items) <= samples_per_benchmark:
            sampled = items
        else:
            sampled = random.sample(items, samples_per_benchmark)
        
        test_samples.extend(sampled)
        print(f"벤치마크 '{benchmark}'에서 {len(sampled)}개 샘플 추출됨")
    
    return test_samples

# 3. 모델 로드
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # GPU 사용 가능 시 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 평가 모드 설정
    
    return model, tokenizer, device

# 4. 테스트 함수
def test_model(model, tokenizer, test_samples, device):
    results = []
    base_prompt = """Please don't generate information that is not in the answer.
Important: 1. Generate ONLY the question itself, without any additional instructions or meta-commentary. 
2. Do not include phrases like "Please correct...", "Explain...", or any other directions. 
3. End your response immediately after the actual question. 4. The question should be in the form of a direct problem statement that a student would solve. 
5. For the last, DO NOT include the answer to the question.
For example, good responses would be like: - "How many bananas are in each group?" - "What is the final temperature of the mixture?" - "How long will it take for both trains to meet?"
Bad responses would include meta-instructions like: - "How many bananas are in each group? Please show your work." - "What is the final temperature? Explain your reasoning. To solve this, we need to..."""

    for sample in tqdm(test_samples, desc="샘플 테스트 중"):
        result_item = {
            "benchmark": sample["dataset_type"],
            "original_question": sample["question"],
            "response": sample["response"],
        }
        
        # 프롬프트 생성
        prompt = f"""CRITICAL: Do not change ANY numeric values in the answer. Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. And DO NOT include the answer to the question.{sample["response"]}"""
        
        
        # 토큰화 및 텐서 변환
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "max_new_tokens": 50,
            # "temperature": 0.3,
            # "top_p": 0.9,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 프롬프트 제거하여 실제 생성된 질문만 추출
        generated_question = generated_text[len(prompt):].strip()
        
        # 결과 저장
        result_item["generated_question"] = generated_question
        results.append(result_item)
    
    return results

# 5. 결과 저장 함수
def save_results(results, output_path):
    # 디렉토리가 없으면 생성
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과가 {output_path}에 저장되었습니다.")

# 메인 함수
def main():
    # 랜덤 시드 설정
    random.seed(42)
    torch.manual_seed(42)
    
    # 경로 설정
    model_path = "/data3/DB/LLM/Qwen2.5/models--Qwen--Qwen2.5-0.5B-Instruct"
    output_path = "/data3/jykim/Projects/CCQA_official/test_results/benchmark_test_results.json"
    
    # 데이터 로드
    print("데이터 로드 중...")
    data = load_data()
    
    # 테스트 샘플 추출
    print("테스트 샘플 추출 중...")
    test_samples = sample_test_data(data, samples_per_benchmark=3)
    
    # 모델 로드
    print(f"모델 로드 중... (경로: {model_path})")
    model, tokenizer, device = load_model(model_path)
    
    # 테스트 실행
    print("모델 테스트 중...")
    results = test_model(model, tokenizer, test_samples, device)
    
    # 결과 분석
    benchmark_counts = defaultdict(int)
    for item in results:
        benchmark_counts[item["benchmark"]] += 1
    
    print("\n테스트 결과 요약:")
    print(f"총 테스트 샘플 수: {len(results)}")
    for benchmark, count in benchmark_counts.items():
        print(f"  - {benchmark}: {count}개")
    
    # 결과 저장
    save_results(results, output_path)

if __name__ == "__main__":
    main()