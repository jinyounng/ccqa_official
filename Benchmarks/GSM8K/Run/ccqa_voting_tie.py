# -*- coding: utf-8 -*-
"""
 1) 정답 추출 패턴 4‑가지 그대로 유지
 2) 가중치 투표 → 과반수 미만이면 cosine 최고점 폴백
 3) INPUT_ROOT 아래 모든 *.json (재귀) 처리
"""

import os, glob, json, re, csv
from collections import Counter
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ────────────────────────────────────────────────────────
# 1.  정답 추출 (기존 패턴 유지)
# ────────────────────────────────────────────────────────
ANSWER_PATTERNS = [
    r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)',
    r'\b(?:so\s+)?the\s+(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-Ea-e])\)?',
    r'\banswer\s*:?\s*\(?([A-Ea-e])\)?',
]
def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower()
    for pat in ANSWER_PATTERNS:
        m = re.search(pat, low)
        if m:
            return m.group(1).replace(',', '').strip()
    return None

def numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return (a or '').strip().lower() == (b or '').strip().lower()

# ────────────────────────────────────────────────────────
# 2.  가중치 투표 + cosine 폴백
# ────────────────────────────────────────────────────────
def vote_with_cosine(item: Dict[str, Any], weights: List[int]) -> Optional[str]:
    idxs = item.get("most_similar_idxs", [])[:len(weights)]
    if len(idxs) < len(weights):
        idxs += [i for i in range(1, 6) if i not in idxs][:len(weights) - len(idxs)]

    votes, cos_map = [], {}
    for w, idx in zip(weights, idxs):
        resp = item.get(f"response_{idx}", "")
        ans  = extract_answer(resp)
        if not ans:
            continue
        votes.extend([ans] * w)

        cos = item.get("similarity_scores", {}).get(f"question_{idx}", {}).get("cosine", 0.0)
        cos_map[ans] = max(cos_map.get(ans, 0.0), cos)

    if not votes:
        return None

    cnt = Counter(votes)
    best_ans, best_votes = cnt.most_common(1)[0]

    # 과반수면 그대로, 아니면 cosine 최고점
    return best_ans if best_votes > len(votes) // 2 else max(cos_map, key=cos_map.get)

# ────────────────────────────────────────────────────────
# 3.  개별 JSON 파일 처리
# ────────────────────────────────────────────────────────
def process_file(path: str, weights: List[int]) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data

    total = len(results)
    correct = 0
    for item in results:
        pred = vote_with_cosine(item, weights)
        item["final_answer"] = pred
        if pred and numeric_equal(pred, item.get("correct_answer")):
            correct += 1

    return {
        "file": os.path.relpath(path, start=INPUT_ROOT),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0
    }

# ────────────────────────────────────────────────────────
# 4.  메인 : ROOT 하위의 모든 JSON (재귀)
# ────────────────────────────────────────────────────────
INPUT_ROOT = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate"    
WEIGHTS    = [5, 4, 4, 3, 2]           
OUTPUT_CSV = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate/summary_all_files.csv"

def main():
    json_paths = glob.glob(os.path.join(INPUT_ROOT, "**", "*.json"), recursive=True)
    if not json_paths:
        print(" JSON 파일을 찾지 못했습니다."); return

    print(f"총 {len(json_paths)}개 JSON 처리 중…")
    rows = [process_file(fp, WEIGHTS) for fp in tqdm(json_paths, leave=False)]

    # CSV 저장
    with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["file", "total", "correct", "accuracy"])
        w.writeheader(); w.writerows(rows)

    overall_acc = sum(r["correct"] for r in rows) / sum(r["total"] for r in rows)
    print(f"\n 완료. 전체 정확도 {overall_acc:.4%}  |  결과 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
