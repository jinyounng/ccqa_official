import os
import json
import glob
import torch
import pickle
from tqdm import tqdm
from bert_score import BERTScorer
from collections import defaultdict
import traceback

# === 경로 설정 ===
input_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_0_result"
output_directory = os.path.join(input_directory, "F1_similar")
idf_weights_file = "/data3/jykim/Projects/CCQA_official/bert_idf_weights.pkl"

os.makedirs(output_directory, exist_ok=True)

# === IDF 가중치 로드 + 안전 처리 ===
print(f"Loading IDF weights from {idf_weights_file}...")
if not os.path.exists(idf_weights_file):
    raise FileNotFoundError(f"IDF weights file not found: {idf_weights_file}")

with open(idf_weights_file, "rb") as f:
    idf_dict = pickle.load(f)

idf_dict = defaultdict(lambda: 1.0, idf_dict)  # 없는 토큰엔 기본값 1.0

# === BERTScorer 초기화 및 IDF 적용 ===
print("Initializing BERTScorer with IDF weights...")
scorer = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    idf=True
)
scorer._idf_dict = idf_dict
print("BERTScorer ready.")

# === JSON 파일 처리 ===
json_files = glob.glob(os.path.join(input_directory, "*.json"))
print(f"Found {len(json_files)} files")

for json_file in tqdm(json_files, desc="Processing files"):
    filename = os.path.basename(json_file)
    output_file = os.path.join(output_directory, filename)

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'results' not in data:
            tqdm.write(f"[SKIP] '{filename}' - No 'results' key")
            continue

        for i, result in enumerate(tqdm(data['results'], desc=f"{filename}", leave=False)):
            original_question = result.get('question', '').strip()
            if not original_question:
                continue

            generated_questions = [
                result.get(f"generated_question_{j}", "").strip()
                for j in range(1, 6)
                if result.get(f"generated_question_{j}", "").strip()
            ]
            keys = [j for j in range(1, 6) if result.get(f"generated_question_{j}", "").strip()]

            if not generated_questions:
                continue

            similarities = []
            batch_size = 8
            for start in range(0, len(generated_questions), batch_size):
                end = start + batch_size
                batch = generated_questions[start:end]
                batch_keys = keys[start:end]

                _, F1, _ = scorer.score([original_question] * len(batch), batch)
                similarities.extend([
                    (batch_keys[j], F1[j].item())
                    for j in range(len(batch))
                ])

            similarities.sort(key=lambda x: x[1], reverse=True)
            result['most_similar_idxs'] = [idx for idx, _ in similarities]
            result['similarity_scores'] = {f"question_{idx}": float(sim) for idx, sim in similarities}

        # 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        tqdm.write(f"[DONE] {filename} → saved")

    except Exception as e:
        tqdm.write(f"[ERROR] {filename}: {e}")
        traceback.print_exc()

print("✅ All files processed.")
