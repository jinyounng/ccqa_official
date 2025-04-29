import os
import json
import pandas as pd

# 폴더 경로 설정
folder_path = 'Benchmarks/CommonsenseQA/Results/ccqa_result/similarity_results'

# 결과를 저장할 리스트 초기화
results = []

# 폴더 내의 모든 파일을 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # 각 파일별 정답 맞춘 횟수와 전체 문제 수를 저장할 변수 초기화
        correct_count = 0
        total_questions = 0
        
        # JSON 파일 열기 및 데이터 로드
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # 각 데이터 항목에 대해 정답 비교
        for entry in data:
            answer_key = entry.get('answerKey')
            extracted_answer = entry.get('extracted_answer')
            
            # 정답 비교 후 카운트 증가
            if answer_key == extracted_answer:
                correct_count += 1
            total_questions += 1

        # 파일별 정확도 계산
        accuracy = correct_count / total_questions if total_questions else 0

        # 결과 저장
        results.append({
            'Filename': filename,
            'Total Questions': total_questions,
            'Correct Answers': correct_count,
            'Accuracy': accuracy
        })

# 결과를 DataFrame으로 변환
df = pd.DataFrame(results)

# CSV 파일로 저장
csv_file_path = os.path.join(folder_path, 'file_accuracy_results.csv')
df.to_csv(csv_file_path, index=False)

print(f"결과가 저장된 파일: {csv_file_path}")
