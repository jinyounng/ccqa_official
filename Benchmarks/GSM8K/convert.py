import os
import json

# 처리할 폴더 경로
base_path = "/data3/jykim/Projects/CCQA_official/finetuning/train_set/commonsenseqa_train"

# 폴더 내 모든 JSON 파일을 순회
for filename in os.listdir(base_path):
    if filename.endswith(".json"):
        file_path = os.path.join(base_path, filename)

        # 파일 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing {filename}: {e}")
                continue

        # 변환이 필요한지 확인하기 위한 플래그
        needs_conversion = False
        
        # Assume data is a list of results
        for result in data:
            # all_responses가 있는 경우에만 변환
            if "all_responses" in result and isinstance(result["all_responses"], list):
                needs_conversion = True
                all_responses = result.pop("all_responses", [])
                for i, response in enumerate(all_responses, start=1):
                    result[f"response_{i}"] = response
        
        # 변환이 필요한 경우에만 파일 덮어쓰기
        if needs_conversion:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Converted: {filename}")
        else:
            print(f"No conversion needed: {filename}")

print("All files processed.")