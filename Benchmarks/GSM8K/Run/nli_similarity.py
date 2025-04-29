import os
import json
import glob
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import traceback

# ===== 경로 설정 =====
input_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result"
output_directory = os.path.join(input_directory, "precision_nli_separate")
os.makedirs(output_directory, exist_ok=True)


# ===== 모델 초기화 =====
print("Initializing models...")
nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=0 if torch.cuda.is_available() else -1)
bert_scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, idf=False)
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # ~22M
contrastive_model = SentenceTransformer("BAAI/bge-large-en-v1.5")  # ~400M
print("All models initialized.\n")

# ===== 파일 처리 =====
json_files = glob.glob(os.path.join(input_directory, "*.json"))
print(f"Found {len(json_files)} JSON files")

for json_file in tqdm(json_files, desc="Processing files"):
    filename = os.path.basename(json_file)
    output_file = os.path.join(output_directory, filename)

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'results' not in data:
            tqdm.write(f"Skipping {filename}: 'results' key not found")
            continue

        for i, result in enumerate(tqdm(data['results'], desc=f"{filename}", leave=False)):
            original_question = result.get('question', '').strip()
            if not original_question:
                continue

            # 생성된 질문 추출
            generated_questions = []
            generated_question_keys = []

            for j in range(1, 6):
                key = f'generated_question_{j}'
                if key in result and result[key]:
                    generated_questions.append(result[key])
                    generated_question_keys.append(j)

            if generated_questions:
                # ===== 1. BERTScore Precision =====
                P, _, _ = bert_scorer.score([original_question] * len(generated_questions), generated_questions)
                precision_scores = [p.item() for p in P]

                # ===== 2. NLI entailment scores =====
                entailment_scores = []
                for gen_q in generated_questions:
                    backward_text = f"{original_question} </s> {gen_q}"
                    forward_text = f"{gen_q} </s> {original_question}"

                    backward_res = nli_pipeline(backward_text)
                    forward_res = nli_pipeline(forward_text)

                    backward_prob = next((r['score'] for r in backward_res if r['label'].upper() == 'ENTAILMENT'), 0.0)
                    forward_prob = next((r['score'] for r in forward_res if r['label'].upper() == 'ENTAILMENT'), 0.0)

                    entailment_scores.append((forward_prob, backward_prob))

                # ===== 3. Cosine similarity (SBERT) =====
                all_sentences = [original_question] + generated_questions
                embeddings = sbert_model.encode(all_sentences, convert_to_tensor=True)
                cosine_scores = util.cos_sim(embeddings[0], embeddings[1:]).squeeze().tolist()

                # ===== 4. Contrastive similarity (BGE) =====
                contrastive_embeddings = contrastive_model.encode(all_sentences, convert_to_tensor=True, normalize_embeddings=True)
                contrastive_scores = util.cos_sim(contrastive_embeddings[0], contrastive_embeddings[1:]).squeeze().tolist()

                # ===== 5. 저장 =====
                result['similarity_scores'] = {
                    f"question_{generated_question_keys[i]}": {
                        "precision": round(precision_scores[i], 4),
                        "entailment_forward": round(entailment_scores[i][0], 4),
                        "entailment_backward": round(entailment_scores[i][1], 4),
                        "entailment_min": round(min(entailment_scores[i]), 4),
                        "cosine": round(cosine_scores[i], 4),
                        "contrastive": round(contrastive_scores[i], 4)
                    }
                    for i in range(len(generated_question_keys))
                }

                # ===== 6. 정렬 기준 (Precision만 사용 중) =====
                precision_scores_with_index = [
                    (generated_question_keys[i], precision_scores[i])
                    for i in range(len(generated_question_keys))
                ]
                precision_scores_with_index.sort(key=lambda x: x[1], reverse=True)
                result['most_similar_idxs'] = [idx for idx, _ in precision_scores_with_index]

        # ===== 결과 저장 =====
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        tqdm.write(f"Saved → {output_file}")

    except Exception as e:
        tqdm.write(f"[ERROR] {filename}: {e}")
        traceback.print_exc()

print("\n All processing completed!")