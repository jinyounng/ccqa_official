import os
import json
import glob
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# 디렉토리 경로
input_directory = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results"
output_directory = "/home/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/bleu_similarity"

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# JSON 파일 목록 가져오기
json_files = glob.glob(os.path.join(input_directory, "*.json"))
print(f"Found {len(json_files)} JSON files")

# BLEU 스코어 계산 함수
def calculate_bleu(reference, candidate):
    """
    BLEU 스코어를 계산하는 함수
    reference: 참조 문장 (원본 질문)
    candidate: 후보 문장 (생성된 질문)
    """
    # 문장을 토큰화
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    # weights 설정 (unigram만 고려) - 단순 단어 중복에 초점
    weights = (1, 0, 0, 0)
    
    # 참조가 빈 경우 처리
    if not reference_tokens:
        return 0.0
    
    # BLEU 스코어 계산
    try:
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights)
    except ZeroDivisionError:
        # 분모가 0이 되는 경우에 대한 예외 처리
        return 0.0
    
    return bleu_score

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
            
            # 생성된 질문 추출 - regenerated_question_n 키 사용
            generated_questions = []
            generated_question_keys = []
            
            for j in range(1, 6):  # 생성된 질문 5개 가정
                key = f'regenerated_question_{j}'  # 수정된 키 이름
                if key in result and result[key]:
                    generated_questions.append(result[key])
                    generated_question_keys.append(j)
            
            # 생성된 질문이 있는 경우 유사도 계산
            if generated_questions:
                similarities = []
                
                # BLEU 스코어를 사용한 유사도 계산
                for j, question in enumerate(generated_questions):
                    bleu_score = calculate_bleu(original_question, question)
                    similarities.append((generated_question_keys[j], bleu_score))
                
                # 유사도를 기준으로 내림차순 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 유사도 순서대로 인덱스 추출
                most_similar_idxs = [idx for idx, _ in similarities]
                
                # 기존 most_similar_idx 확인 및 디버깅
                if 'most_similar_idx' in result:
                    old_idx = result['most_similar_idx']
                    if most_similar_idxs and old_idx != most_similar_idxs[0]:
                        tqdm.write(f"Item {i}: Old idx: {old_idx}, New first idx: {most_similar_idxs[0]}")
                
                # 결과에 추가
                result['most_similar_idxs'] = most_similar_idxs
                
                # 디버깅을 위해 유사도 점수도 저장
                result['similarity_scores'] = {f"question_{idx}": float(sim) for idx, sim in similarities}
                
                # 가장 유사도가 높은 질문 전체 텍스트도 저장
                if most_similar_idxs:
                    most_similar_idx = most_similar_idxs[0]
                    most_similar_question = result.get(f'regenerated_question_{most_similar_idx}', '')
                    result['most_similar_question_text'] = most_similar_question
            
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