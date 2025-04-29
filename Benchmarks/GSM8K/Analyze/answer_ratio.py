import json
import re
import os
import csv
from collections import defaultdict, Counter
from tqdm import tqdm

def extract_numerical_answer(response):
    """답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자를 추출하는 함수"""
    if not response:
        return None
    
    # Common answer patterns
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    
    # 텍스트를 소문자로 변환
    text = response.lower()
    
    # 패턴들을 순차적으로 검사
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # 캡처된 그룹(숫자) 추출
            answer = match.group(1).strip()
            # 숫자에 천 단위 구분 콤마(,)가 있으면 제거
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    
    return None

def is_answer_correct(extracted_answer, correct_answer):
    """추출된 답변과 정답이 일치하는지 확인하는 함수"""
    if extracted_answer is None or correct_answer is None:
        return False
    
    # 숫자만 추출하여 비교
    extracted_numeric = re.sub(r'[^\d.]', '', str(extracted_answer).strip())
    correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
    
    try:
        if extracted_numeric and correct_numeric:
            extracted_float = float(extracted_numeric)
            correct_float = float(correct_numeric)
            return abs(extracted_float - correct_float) < 1e-5
        else:
            # 숫자가 아닌 경우 문자열 비교
            return str(extracted_answer).strip() == str(correct_answer).strip()
    except ValueError:
        # 숫자로 변환할 수 없는 경우 단순 문자열 비교
        return str(extracted_answer).strip() == str(correct_answer).strip()

def create_answer_ratio(answer_counts, correct_answer):
    """
    정답 카운트를 맨 앞에 두고, 나머지는 빈도순(내림차순)으로 붙인다.
    정답이 하나도 없으면 0부터 시작.
    예) 정답 2회, 오답 3·1·1회  →  "2:3:1:1"
        정답 없음, 오답 2·1·1·1회 →  "0:2:1:1:1"
    """
    # 정답 카운트 찾기
    correct_count = 0
    for ans, cnt in answer_counts.items():
        if is_answer_correct(ans, correct_answer):
            correct_count = cnt
            break

    # 정답을 제외한 나머지만 정렬
    others = [
        (ans, cnt) for ans, cnt in answer_counts.items()
        if not is_answer_correct(ans, correct_answer)
    ]
    others_sorted = sorted(others, key=lambda x: -x[1])

    # 비율 문자열 생성
    ratio_parts = [str(correct_count)]                    # 무조건 맨 앞
    ratio_parts += [str(cnt) for _, cnt in others_sorted] # 나머지
    return ":".join(ratio_parts)


def analyze_answer_ratios(results):
    """각 문제의 답변 비율을 분석하고 비율별로 카운트"""
    # 비율별 카운트
    ratio_counts = defaultdict(int)
    sc_correct_counts = defaultdict(int)
    sc_wrong_counts = defaultdict(int)
    
    for item in results:
        # 정답이 없으면 건너뜀
        if "correct_answer" not in item:
            continue
        
        correct_answer = item["correct_answer"]
        
        # 응답 목록 생성
        responses = []
        for i in range(1, 6):
            response_key = f"response_{i}"
            if response_key in item and item[response_key]:
                responses.append(item[response_key])
        
        # 각 응답에서 답변 추출
        extracted_answers = []
        for response in responses:
            answer = extract_numerical_answer(response)
            if answer:
                extracted_answers.append(answer)
        
        # 답변 카운트
        answer_counts = Counter(extracted_answers)
        
        # 답변이 없으면 건너뜀
        if not answer_counts:
            continue
        
        # 답변 비율 문자열 생성
        ratio_str = create_answer_ratio(answer_counts, correct_answer)
        
        # 비율 카운트 증가
        ratio_counts[ratio_str] += 1
        
        # Self-Consistency 계산
        # answer_counts에서 최빈값이 여러 개면 정답을 우선 뽑기
        most_common_answer = max(
            answer_counts.items(),
            key=lambda x: (x[1], is_answer_correct(x[0], correct_answer))
        )[0]

        sc_correct = is_answer_correct(most_common_answer, correct_answer)
        
        # Self-Consistency 결과에 따라 비율별 카운트 증가
        if sc_correct:
            sc_correct_counts[ratio_str] += 1
        else:
            sc_wrong_counts[ratio_str] += 1
    
    return {
        "ratio_counts": dict(ratio_counts),
        "sc_correct_counts": dict(sc_correct_counts),
        "sc_wrong_counts": dict(sc_wrong_counts),
    }

def process_all_files(input_folders, output_folder="answer_ratio_results"):
    """여러 폴더의 JSON 파일을 처리하여 모델별로 답변 비율 분포를 CSV 파일로 저장"""
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    for input_folder in input_folders:
        folder_name = os.path.basename(input_folder.rstrip('/'))
        similarity_method = folder_name.replace("_similar", "")
        
        print(f"\n폴더 '{similarity_method}' 처리 중...")
        
        # 모든 JSON 파일 목록 가져오기
        json_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        if not json_files:
            print(f"경고: {input_folder}에서 JSON 파일을 찾을 수 없습니다!")
            continue
        
        print(f"총 {len(json_files)}개 파일을 처리합니다.")
        
        # 각 파일 처리
        for file_path in tqdm(json_files, desc=f"{similarity_method} 파일 처리 중"):
            try:
                # 파일명에서 모델명 추출
                filename = os.path.basename(file_path)
                
                # GSM8K_ModelName_ 패턴에서 모델명 추출
                model_match = re.search(r'GSM8K_([^_]+)_', filename)
                model_name = model_match.group(1) if model_match else os.path.splitext(filename)[0]
                
                # 파일 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 결과 항목 가져오기
                if isinstance(data, dict) and "results" in data:
                    results = data["results"]
                else:
                    results = data
                
                # 답변 비율 분석
                stats = analyze_answer_ratios(results)
                
                # 결과 정렬 및 저장
                ratio_counts = stats["ratio_counts"]
                sorted_ratios = sorted(ratio_counts.items(), key=lambda x: -x[1])
                
                # 모델별 분포 CSV 파일 생성
                output_file = os.path.join(output_folder, f"{similarity_method}_{model_name}_ratio_distribution.csv")
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['ratio', 'count', 'sc_correct', 'sc_wrong', 'sc_accuracy'])
                    
                    for ratio, count in sorted_ratios:
                        sc_correct = stats["sc_correct_counts"].get(ratio, 0)
                        sc_wrong = stats["sc_wrong_counts"].get(ratio, 0)
                        sc_accuracy = round(sc_correct / count, 4) if count > 0 else 0
                        
                        writer.writerow([ratio, count, sc_correct, sc_wrong, sc_accuracy])
                
                print(f"모델 {model_name}의 비율 분포가 {output_file}에 저장되었습니다.")
                
            except Exception as e:
                print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n모든 모델의 비율 분포 저장이 완료되었습니다.")

def main():
    # 입력 폴더 목록
    INPUT_FOLDERS = [
        "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate",
    ]
    
    # 출력 폴더
    OUTPUT_FOLDER = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Analyze/answer_ratio_by_model"
    
    # 모든 파일 처리하고 모델별로 CSV 저장
    process_all_files(INPUT_FOLDERS, OUTPUT_FOLDER)
    
    print("처리가 완료되었습니다.")

if __name__ == "__main__":
    main()