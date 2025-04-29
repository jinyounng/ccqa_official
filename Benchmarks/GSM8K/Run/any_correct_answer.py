import json
import re
import os
from collections import defaultdict, Counter
from tqdm import tqdm

def extract_numerical_answer(response):
    """
    답변에서 처음 등장하는 'the answer is' 패턴을 찾아 숫자를 추출하는 함수
    """
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
    """
    추출된 답변과 정답이 일치하는지 확인하는 함수
    """
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

def get_ccqa_answer(item):
    """
    CCQA 방식을 사용하여 최종 답변을 계산
    
    - most_similar_idxs에 따라 가중치 [5,5,4,4,3] 적용
    - 답변 추출 후 가중 투표
    """
    if "most_similar_idxs" not in item or not item["most_similar_idxs"]:
        return None
    
    # 가중치 정의
    weights = [5, 5, 4, 4, 3]
    
    # 상위 N개 인덱스 가져오기 (순서대로)
    top_indices = item["most_similar_idxs"][:len(weights)]
    
    # 각 응답에서 정답 추출 후 가중 투표
    answer_votes = []
    
    for idx, resp_idx in enumerate(top_indices):
        if 1 <= resp_idx <= 5:  # 유효한 응답 인덱스인지 확인
            response_key = f"response_{resp_idx}"
            if response_key in item and item[response_key]:
                answer = extract_numerical_answer(item[response_key])
                if answer:
                    weight = weights[idx]
                    answer_votes.extend([answer] * weight)
    
    # 투표 결과가 있으면 다수결로 결정
    if answer_votes:
        vote_counts = Counter(answer_votes)
        most_common = vote_counts.most_common(1)
        if most_common:
            return most_common[0][0]
    
    return None

def filter_by_correct_count(results, output_folder, model_name, similarity_method, target_counts=[1, 2]):
    """
    정답 개수(1개 또는 2개 정확히)에 따라 문제를 필터링하고 
    CCQA 방식으로 틀린 경우만 저장하는 함수
    """
    # 결과를 저장할 딕셔너리 초기화
    filtered_results = {count: [] for count in target_counts}
    
    # 각 항목 처리
    for item in results:
        # 정답이 없으면 건너뜀
        if "correct_answer" not in item:
            continue
        
        correct_answer = item["correct_answer"]
        
        # 각 응답(1~5)이 정답인지 확인하고 카운트
        correct_count = 0
        individual_correct = []
        
        for i in range(1, 6):
            response_key = f"response_{i}"
            if response_key not in item or not item[response_key]:
                individual_correct.append(False)
                continue
            
            # 응답에서 답변 추출
            extracted_answer = extract_numerical_answer(item[response_key])
            
            # 추출된 답변이 정답과 일치하는지 확인
            is_correct = is_answer_correct(extracted_answer, correct_answer)
            individual_correct.append(is_correct)
            
            if is_correct:
                correct_count += 1
        
        # 정확히 target_counts 개수만큼 정답인 경우만 필터링
        if correct_count in target_counts:
            # CCQA 답변 계산
            ccqa_answer = get_ccqa_answer(item)
            # CCQA 답변이 틀린 경우만 저장
            if not is_answer_correct(ccqa_answer, correct_answer):
                filtered_results[correct_count].append(item)
    
    # 결과 저장
    for count in target_counts:
        # 폴더가 없으면 생성
        count_folder = os.path.join(output_folder, f"exact_{count}_correct_ccqa_wrong")
        os.makedirs(count_folder, exist_ok=True)
        
        # 파일 저장
        output_file = os.path.join(count_folder, f"{similarity_method}_{model_name}_exact_{count}_correct_ccqa_wrong.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            output_data = {
                "method": similarity_method,
                "model": model_name,
                "exact_correct_count": count,
                "ccqa_result": "wrong",
                "total_items": len(filtered_results[count]),
                "results": filtered_results[count]
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"정확히 {count}개 정답이고 CCQA가 틀린 문제 {len(filtered_results[count])}개를 {output_file}에 저장했습니다.")
    
    # 통계 반환
    stats = {count: len(filtered_results[count]) for count in target_counts}
    return stats

def process_all_files(input_folders, output_root_folder="filtered_by_correct_count_ccqa_wrong"):
    """
    여러 폴더의 JSON 파일을 처리하여 정답 개수별로 필터링하고
    CCQA가 틀린 문제만 저장
    """
    # 출력 폴더 생성
    os.makedirs(output_root_folder, exist_ok=True)
    
    all_stats = {}
    
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
        
        folder_stats = {}
        
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
                
                # 정답 개수별로 필터링하고 CCQA가 틀린 경우만 저장
                model_stats = filter_by_correct_count(
                    results, 
                    output_root_folder, 
                    model_name, 
                    similarity_method,
                    target_counts=[1, 2]  # 정확히 1개 또는 2개만 정답인 경우
                )
                
                folder_stats[model_name] = model_stats
                
            except Exception as e:
                print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        all_stats[similarity_method] = folder_stats
    
    # 통계 내보내기
    output_stats_file = os.path.join(output_root_folder, "filtering_stats_ccqa_wrong.json")
    with open(output_stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n필터링 통계가 {output_stats_file}에 저장되었습니다.")
    
    return all_stats

def print_stats(all_stats):
    """필터링 통계 출력"""
    print("\n--- CCQA가 틀린 문제 필터링 통계 ---")
    
    for similarity_method, folder_stats in all_stats.items():
        print(f"\n유사도 방법: {similarity_method}")
        print(f"{'모델':<20} {'정확히 1개 정답(CCQA 틀림)':<30} {'정확히 2개 정답(CCQA 틀림)':<30}")
        print("-" * 80)
        
        for model_name, stats in folder_stats.items():
            count_1 = stats.get(1, 0)
            count_2 = stats.get(2, 0)
            print(f"{model_name:<20} {count_1:<30} {count_2:<30}")

# 메인 함수
def main():
    # 입력 폴더 목록
    INPUT_FOLDERS = [
        "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate",
    ]
    
    # 출력 폴더 (새로운 폴더에 저장)
    OUTPUT_ROOT_FOLDER = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/filtered_by_correct_count_ccqa_wrong"
    
    # 모든 파일 처리하고 필터링하여 CCQA가 틀린 문제만 저장
    all_stats = process_all_files(INPUT_FOLDERS, OUTPUT_ROOT_FOLDER)
    
    # 통계 출력
    print_stats(all_stats)

if __name__ == "__main__":
    main()