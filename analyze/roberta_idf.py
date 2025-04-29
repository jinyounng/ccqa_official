import os
import json
import pickle
from tqdm import tqdm
from bert_score import BERTScorer

# ---------- 경로 ----------
input_file  = "/data3/jykim/Projects/CCQA_official/finetuning/combined_qa_dataset.json"
output_file = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/bert_idf_weights.pkl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ---------- 데이터 로드 ----------
print(f"Loading data from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} entries")

# ---------- 텍스트 추출 ----------
print("Extracting output text data for IDF calculation...")
text_samples = []
for item in tqdm(data, desc="Extracting output texts"):
    if isinstance(item, dict) and item.get("output", "").strip():
        text_samples.append(item["output"].strip())
    if len(text_samples) >= 10_000:
        print("Collected 10 000 text samples (cap reached)")
        break

if len(text_samples) < 10:  # 안전장치
    text_samples.extend([
        "What is the primary function of the heart?",
        "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
        "Sammy wanted to go to where the people were. Where might he go?",
        "To locate a choker not located in a jewelry box or boutique where would you go?"
    ])

print(f"Extracted {len(text_samples)} text samples")

# ---------- BERTScorer 초기화 & IDF 계산 ----------
print("Initializing BERTScorer and calculating IDF weights...")
scorer = BERTScorer(model_type="roberta-large",
                    lang="en",
                    idf=True,
                    idf_sents=text_samples)

# ---------- IDF 딕셔너리 저장 ----------
idf_dict = dict(scorer._idf_dict)        # defaultdict → 일반 dict
print(f"Saving IDF weights to {output_file}...")
with open(output_file, "wb") as f:
    pickle.dump(idf_dict, f)

print("IDF weights saved successfully!")

# ---------- 예시 출력 ----------
print("\nExample IDF weights:")
for token, weight in list(idf_dict.items())[:10]:
    print(f"Token: {token:<12}  IDF: {weight:.4f}")
