import os
import json
import glob
from collections import defaultdict
from tqdm import tqdm

def extract_answer(text):
    if not text:
        return None
    text = text.lower()
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    import re
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None

def get_ranks(item, correct_answer):
    similarity = item.get("similarity_scores", {})
    correct_ranks = []

    all_scores = {"cosine": [], "entailment": [], "precision": [], "contrastive": []}
    correct_scores = {"cosine": [], "entailment": [], "precision": [], "contrastive": []}

    for i in range(1, 6):
        resp = item.get(f"response_{i}")
        if not resp:
            continue
        ans = extract_answer(resp)
        scores = similarity.get(f"question_{i}", {})
        cos, ent, pre, con = scores.get("cosine", 0), scores.get("entailment_min", 0), scores.get("precision", 0), scores.get("contrastive", 0)
        all_scores["cosine"].append((i, cos))
        all_scores["entailment"].append((i, ent))
        all_scores["precision"].append((i, pre))
        all_scores["contrastive"].append((i, con))
        if ans == correct_answer:
            correct_scores["cosine"].append((i, cos))
            correct_scores["entailment"].append((i, ent))
            correct_scores["precision"].append((i, pre))
            correct_scores["contrastive"].append((i, con))

    result = {}
    for metric in ["cosine", "entailment", "precision", "contrastive"]:
        all_sorted = sorted(all_scores[metric], key=lambda x: x[1], reverse=True)
        idx_to_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(all_sorted)}
        ranks = [idx_to_rank[idx] for (idx, _) in correct_scores[metric]]
        result[metric] = ranks

    return result

def analyze_rank_for_files(folder):
    json_files = glob.glob(os.path.join(folder, "*.json"))
    total = defaultdict(list)

    for file_path in tqdm(json_files, desc=f"Analyzing {folder}"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        results = data["results"] if isinstance(data, dict) else data

        for item in results:
            correct_answer = item.get("answerKey") or item.get("correct_answer")
            if not correct_answer:
                continue
            answer_counts = []
            for i in range(1, 6):
                ans = extract_answer(item.get(f"response_{i}"))
                if ans:
                    answer_counts.append(ans)
            correct_count = answer_counts.count(correct_answer)
            if correct_count in [1, 2]:
                ranks = get_ranks(item, correct_answer)
                for metric in ranks:
                    total[metric].extend(ranks[metric])

    avg_ranks = {k: sum(v)/len(v) if v else 0 for k, v in total.items()}
    return avg_ranks

# 경로 설정
paths = [
    "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_0_result/precision_nli_separate",
    "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_precision_nli_scored",
]

# 실행
for path in paths:
    result = analyze_rank_for_files(path)
    print(f"\n[ 평균 랭크 - {path.split('/')[-1]} ]")
    for metric, avg_rank in result.items():
        print(f"  {metric}: {avg_rank:.2f}")
