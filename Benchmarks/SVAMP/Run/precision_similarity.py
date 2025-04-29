import os
import json
import glob
import torch
from bert_score import BERTScorer
import numpy as np
from tqdm import tqdm
import time

# 디렉토리 경로
input_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result"
output_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/precision_similar"

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# BERTScorer 초기화 - idf=False로 설정하여 IDF 가중치 계산 오류 방지
print("Initializing BERTScorer...")
scorer = BERTScorer(lang="en", rescale_with_baseline=True, idf=False)

# JSON 파일 목록 가져오기
json_files = glob.glob(os.path.join(input_directory, "*.json"))
print(f"Found {len(json_files)} JSON files")

# 각 파일 처리
for json_file in tqdm(json_files, desc="Processing files"):
    filename = os.path.basename(json_file)
    output_file = os.path.join(output_directory, filename)
    
    try:
        # JSON 데이터 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 'results' 키가 있는지 확인
        if 'results' not in data:
            tqdm.write(f"Skipping {filename}: 'results' key not found")
            continue
        
        # 결과 개수 확인
        results_count = len(data['results'])
        tqdm.write(f"{filename}: Processing {results_count} results")
        
        # 각 결과 처리
        for i, result in enumerate(tqdm(data['results'], desc=f"Processing {filename}", leave=False)):
            # 원본 질문 추출
            original_question = result.get('question', '')
            if not original_question:
                continue
            
            # 생성된 질문 추출
            generated_questions = []
            generated_question_keys = []
            
            for j in range(1, 6):  # 생성된 질문 5개 가정
                key = f'generated_question_{j}'
                if key in result and result[key]:
                    generated_questions.append(result[key])
                    generated_question_keys.append(j)
            
            # 생성된 질문이 있는 경우 유사도 계산
            if generated_questions:
                similarities = []
                
                # BERTScore를 사용한 유사도 계산
                P, R, F1 = scorer.score([original_question] * len(generated_questions), generated_questions)
                
                # 각 생성된 질문과의 BERTScore F1 유사도 저장
                for j in range(len(generated_questions)):
                    sim = P[j].item()  # 텐서에서 Python 숫자로 변환
                    similarities.append((generated_question_keys[j], sim))
                
                # 유사도를 기준으로 내림차순 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 유사도 순서대로 인덱스 추출
                most_similar_idxs = [idx for idx, _ in similarities]
                
                # 결과에 추가
                result['most_similar_idxs'] = most_similar_idxs
                
                # 디버깅을 위해 유사도 점수도 저장
                result['similarity_scores'] = {f"question_{idx}": float(sim) for idx, sim in similarities}
            
            # 10개의 결과마다 중간 저장
            if (i + 1) % 10 == 0 or i == results_count - 1:
                tqdm.write(f"Saving intermediate results at item {i+1}/{results_count}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
        
        # 최종 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        tqdm.write(f"Successfully saved to {output_file}")
        
    except Exception as e:
        tqdm.write(f"Error processing {json_file}: {e}")
        import traceback
        traceback.print_exc()  # 상세한 오류 정보 출력

print("Processing complete!")