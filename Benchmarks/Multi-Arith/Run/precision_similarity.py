import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from bert_score import BERTScorer

# 입력 디렉토리
input_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/Multi-Arith/Result/ccqa_multiArith_few_shot_result"

# NLI 파이프라인 초기화
print("Initializing NLI model...")
nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=0 if torch.cuda.is_available() else -1)

# BERTScorer 초기화
print("Initializing BERTScorer...")
scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, idf=False)

# JSON 파일 목록 가져오기
json_files = glob.glob(os.path.join(input_directory, "*.json"))
print(f"Found {len(json_files)} JSON files")

# 결과 저장 디렉토리
output_directory = os.path.join(input_directory, "precision_nli_separate")
os.makedirs(output_directory, exist_ok=True)

for json_file in tqdm(json_files, desc="Processing files"):
    filename = os.path.basename(json_file)
    output_file = os.path.join(output_directory, filename)

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 형식 확인 - 리스트인지 또는 'results' 키가 있는 딕셔너리인지
        if isinstance(data, dict) and 'results' in data:
            items = data['results']
            has_results_key = True
        else:
            # 데이터가 직접 리스트인 경우
            items = data if isinstance(data, list) else [data]
            has_results_key = False
        
        print(f"Processing {filename}: {'with results key' if has_results_key else 'direct list'}, {len(items)} items")
        
        for i, item in enumerate(tqdm(items, desc=f"Processing {filename}", leave=False)):
            original_question = item.get('question', '')
            if not original_question:
                continue

            generated_questions = []
            generated_question_keys = []

            for j in range(1, 6):  # 최대 20개까지 확인
                key = f'generated_question_{j}'
                if key in item and item[key]:
                    generated_questions.append(item[key])
                    generated_question_keys.append(j)

            if generated_questions:
                # 1. Precision score 계산
                P, _, _ = scorer.score([original_question] * len(generated_questions), generated_questions)
                precision_scores = [p.item() for p in P]
                
                # 2. 양방향 Entailment score 계산
                entailment_scores = []

                for gen_q in generated_questions:
                    # 역방향: original → gen_q (원본이 생성된 질문을 포함하는지)
                    backward_text = f"{original_question} </s> {gen_q}"
                    backward_res = nli_pipeline(backward_text)
                    backward_prob = next((r['score'] for r in backward_res if r['label'].upper() == 'ENTAILMENT'), 0.0)

                    # 정방향: gen_q → original (생성된 질문이 원본을 포함하는지)
                    forward_text = f"{gen_q} </s> {original_question}"
                    forward_res = nli_pipeline(forward_text)
                    forward_prob = next((r['score'] for r in forward_res if r['label'].upper() == 'ENTAILMENT'), 0.0)

                    entailment_scores.append((forward_prob, backward_prob))

                # 3. 각 질문에 대한 개별 점수 저장
                item['similarity_scores'] = {
                    f"question_{generated_question_keys[i]}": {
                        "precision": round(precision_scores[i], 4),
                        "entailment_forward": round(entailment_scores[i][0], 4),  # 생성된 질문이 원본을 포함하는지
                        "entailment_backward": round(entailment_scores[i][1], 4), # 원본이 생성된 질문을 포함하는지
                        "entailment_max": round(max(entailment_scores[i]), 4)  # 양방향 중 최대값
                    }
                    for i in range(len(generated_question_keys))
                }

                # 4. precision 기준으로 정렬
                precision_scores_with_index = [
                    (generated_question_keys[i], precision_scores[i])
                    for i in range(len(generated_question_keys))
                ]
                
                # 5. precision 점수 기준으로 정렬
                precision_scores_with_index.sort(key=lambda x: x[1], reverse=True)
                
                # 6. 정렬된 인덱스만 추출
                most_similar_idxs = [idx for idx, _ in precision_scores_with_index]
                item['most_similar_idxs'] = most_similar_idxs

            # 주기적으로 저장 (10개 항목마다 또는 마지막 항목 처리 후)
            if (i + 1) % 10 == 0 or i == len(items) - 1:
                tqdm.write(f"Saving intermediate results at item {i+1}/{len(items)}")
                
                # 원래 형식에 맞게 저장
                if has_results_key:
                    data['results'] = items
                else:
                    data = items
                    
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

        # 최종 저장
        if has_results_key:
            data['results'] = items
        else:
            data = items
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        tqdm.write(f"Successfully saved to {output_file}")

    except Exception as e:
        tqdm.write(f"Error processing {json_file}: {e}")
        import traceback
        traceback.print_exc()

print("\nAll processing completed!")