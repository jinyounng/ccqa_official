import json
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tabulate import tabulate

# 파인튜닝된 모델 경로
MODEL_PATH = "/home/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-answer-extractor"

# 데이터셋 경로
DATASET_PATH = "/home/jykim/Projects/CCQA_official/finetuning/extractor_dataset.json"

def load_model_and_tokenizer(model_path):
    """파인튜닝된 모델과 토크나이저를 로드합니다."""
    print(f"모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def load_balanced_test_examples(dataset_path, samples_per_type=3):
    """각 데이터셋 유형별로 지정된 수의 샘플을 로드합니다."""
    print(f"데이터셋 로드 중: {dataset_path}")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        return []
    
    # 데이터셋 유형별로 그룹화
    dataset_groups = {}
    for item in data:
        dataset_type = item.get('dataset', 'Unknown')
        if dataset_type not in dataset_groups:
            dataset_groups[dataset_type] = []
        dataset_groups[dataset_type].append(item)
    
    # 각 데이터셋 유형의 샘플 카운트 출력
    print("\n데이터셋 유형별 샘플 수:")
    for dataset_type, items in dataset_groups.items():
        print(f"{dataset_type}: {len(items)}개")
    
    # 각 유형별로 지정된 수의 샘플 선택
    examples = []
    for dataset_type, items in dataset_groups.items():
        if len(items) == 0:
            print(f"경고: {dataset_type} 유형의 샘플이 없습니다.")
            continue
            
        # 최대 samples_per_type개 또는 가용한 모든 샘플 선택
        num_to_select = min(samples_per_type, len(items))
        selected = random.sample(items, num_to_select)
        examples.extend(selected)
        print(f"{dataset_type}에서 {num_to_select}개 샘플 선택됨")
    
    print(f"총 {len(examples)}개의 테스트 예제 로드됨")
    
    # 선택된 예제 정보 출력
    for i, example in enumerate(examples):
        print(f"예제 {i+1}: {example.get('dataset')} 데이터셋")
    
    return examples

def extract_answer(model, tokenizer, question, generated_answer, dataset_type):
    """모델을 사용하여 생성된 답변에서 정답을 추출합니다."""
    # 데이터셋 유형에 따라 프롬프트 구성
    if dataset_type == 'CommonSenseQA':
        prompt = f"Extract the correct answer from this response. Question: {question} Generated response: {generated_answer}"
    else:  # GSM8K, SVAMP 또는 기타
        prompt = f"Extract the numerical answer from this solution. Question: {question} Solution: {generated_answer}"
    
    # 토큰화 및 모델 실행
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=64, 
        num_beams=4, 
        early_stopping=True
    )
    
    # 결과 디코딩
    extracted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extracted_answer

def run_balanced_tests(samples_per_type=3):
    """파인튜닝된 모델을 사용하여 균형잡힌 테스트를 실행합니다."""
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # 균형잡힌 테스트 예제 로드
    test_examples = load_balanced_test_examples(DATASET_PATH, samples_per_type)
    
    if not test_examples:
        print("테스트 예제를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 결과를 저장할 리스트
    results = []
    
    # 각 예제에 대해 모델 테스트
    for i, example in enumerate(test_examples):
        print(f"\n[{i+1}/{len(test_examples)}] 예제 테스트 중... (데이터셋: {example['dataset']})")
        question = example['question']
        generated_answer = example['generated_answer']
        correct_answer = example['answer']
        dataset_type = example['dataset']
        
        # 모델이 추출한 답변
        extracted_answer = extract_answer(model, tokenizer, question, generated_answer, dataset_type)
        
        # 결과 비교 및 저장
        is_correct = extracted_answer.strip() == correct_answer.strip()
        
        # 결과 저장
        results.append({
            'Example': i+1,
            'Dataset': dataset_type,
            'Question': question[:50] + ('...' if len(question) > 50 else ''),
            'Generated Answer': generated_answer[:50] + ('...' if len(generated_answer) > 50 else ''),
            'Correct Answer': correct_answer,
            'Extracted Answer': extracted_answer,
            'Is Correct': '✓' if is_correct else '✗'
        })
        
        # 콘솔에 결과 출력
        print(f"질문: {question[:100]}...")
        print(f"생성된 답변: {generated_answer[:100]}...")
        print(f"정답: {correct_answer}")
        print(f"추출된 답변: {extracted_answer}")
        print(f"정확도: {'정확' if is_correct else '오류'}")
    
    # 결과를 데이터프레임으로 변환하여 테이블 형태로 출력
    df = pd.DataFrame(results)
    print("\n\n===== 테스트 결과 요약 =====")
    print(tabulate(df[['Example', 'Dataset', 'Correct Answer', 'Extracted Answer', 'Is Correct']], headers='keys', tablefmt='grid'))
    
    # 정확도 계산
    accuracy = sum(1 for r in results if r['Is Correct'] == '✓') / len(results) * 100 if results else 0
    print(f"\n전체 정확도: {accuracy:.2f}%")
    
    # 데이터셋별 정확도 계산
    dataset_accuracy = {}
    for dataset_type in set(r['Dataset'] for r in results):
        dataset_examples = [r for r in results if r['Dataset'] == dataset_type]
        dataset_correct = sum(1 for r in dataset_examples if r['Is Correct'] == '✓')
        dataset_accuracy[dataset_type] = dataset_correct / len(dataset_examples) * 100
    
    print("\n데이터셋별 정확도:")
    for dataset_type, acc in dataset_accuracy.items():
        print(f"{dataset_type}: {acc:.2f}%")
    
    # 결과를 CSV 파일로 저장 (선택사항)
    output_file = "test_results_balanced.csv"
    df.to_csv(output_file, index=False)
    print(f"\n결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    # 각 데이터셋 유형별로 3개의 샘플 테스트 (필요에 따라 조정)
    run_balanced_tests(samples_per_type=3)