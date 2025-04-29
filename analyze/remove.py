import json
import os
import glob

def remove_field_from_json(file_path, field_to_remove):
    """
    JSON 파일에서 특정 필드를 제거합니다.
    
    Args:
        file_path (str): JSON 파일 경로
        field_to_remove (str): 제거할 필드 이름
    """
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return False
    
    try:
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # 데이터 구조 처리
        if isinstance(data, dict) and "results" in data:
            # "results" 키가 있는 경우
            for item in data["results"]:
                if field_to_remove in item:
                    del item[field_to_remove]
                    modified = True
        elif isinstance(data, list):
            # 바로 배열인 경우
            for item in data:
                if field_to_remove in item:
                    del item[field_to_remove]
                    modified = True
        elif isinstance(data, dict):
            # 단일 항목인 경우
            if field_to_remove in data:
                del data[field_to_remove]
                modified = True
        
        if modified:
            # 수정된 데이터를 다시 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"'{field_to_remove}' 필드가 '{file_path}'에서 성공적으로 제거되었습니다.")
            return True
        else:
            print(f"'{field_to_remove}' 필드가 '{file_path}'에 존재하지 않습니다.")
            return False
    
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {str(e)}")
        return False

def process_directory(directory_path, file_pattern, field_to_remove):
    """
    지정된 디렉토리에서 패턴과 일치하는 모든 JSON 파일을 처리합니다.
    
    Args:
        directory_path (str): 처리할 디렉토리 경로
        file_pattern (str): 파일 패턴 (예: "*.json")
        field_to_remove (str): 제거할 필드 이름
    """
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"디렉토리가 존재하지 않습니다: {directory_path}")
        return
    
    # 패턴과 일치하는 모든 파일 찾기
    file_pattern_path = os.path.join(directory_path, file_pattern)
    json_files = glob.glob(file_pattern_path)
    
    if not json_files:
        print(f"'{file_pattern_path}' 패턴과 일치하는 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 파일을 처리합니다...")
    
    success_count = 0
    for file_path in json_files:
        if "time_summary" in file_path:
            print(f"시간 요약 파일 건너뜀: {file_path}")
            continue
            
        if remove_field_from_json(file_path, field_to_remove):
            success_count += 1
    
    print(f"처리 완료: {success_count}/{len(json_files)} 파일이 성공적으로 처리되었습니다.")

# 사용 예시
if __name__ == "__main__":
    directory_path = "/home/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result"
    file_pattern = "*.json"
    field_to_remove = "generated_question_20"
    process_directory(directory_path, file_pattern, field_to_remove)