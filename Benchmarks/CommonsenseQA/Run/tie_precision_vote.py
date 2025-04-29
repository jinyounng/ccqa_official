import os
import json
import re
import glob
from collections import Counter, defaultdict
from tqdm import tqdm
import csv
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

def extract_numerical_answer(response):
    if not response:
        return None
    patterns = [
        r'the (?:correct )?answer is (?:[$\u20ac\uffe5\u00a5\u20a9]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    text = response.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    return None

def is_answer_correct(extracted_answer, correct_answer):
    if extracted_answer is None or correct_answer is None:
        return False
    return str(extracted_answer).strip().lower() == str(correct_answer).strip().lower()

def apply_original_self_consistency(results):
    updated_results = []
    for item in tqdm(results, desc="original self-consistency"):
        updated_item = item.copy()
        responses = [item.get(f"response_{i}") for i in range(1, 6) if item.get(f"response_{i}")]
        if not responses:
            updated_item["original_sc_answer"] = None
            updated_results.append(updated_item)
            continue
        extracted_answers = [extract_numerical_answer(r) for r in responses]
        counts = Counter([a for a in extracted_answers if a])
        top = counts.most_common(1)
        updated_item["original_sc_answer"] = top[0][0] if top else None
        updated_results.append(updated_item)
    return updated_results

def apply_new_self_consistency(results, weights=(0.6, 0.2, 0.2)):
    updated_results = []
    rank_counts = {1: 5, 2: 5, 3: 4, 4: 4, 5: 3}
    for item in tqdm(results, desc="new self-consistency"):
        updated_item = item.copy()
        extracted = []
        for i in range(1, 6):
            r = item.get(f"response_{i}")
            if r:
                a = extract_numerical_answer(r)
                if a:
                    extracted.append((i, a))
        if not extracted:
            updated_item["ccqa_answer"] = None
            updated_item["ccqa_method"] = "none"
            updated_results.append(updated_item)
            continue
        counts = Counter([a for _, a in extracted])
        top_answer, top_count = counts.most_common(1)[0]
        if top_count <= 2:
            scored = []
            for idx, ans in extracted:
                sim = item.get("similarity_scores", {}).get(f"question_{idx}", {})
                cos = sim.get("cosine", 0)
                nli = sim.get("entailment_backward", 0)
                prec = sim.get("precision", 0)
                score = weights[0]*cos + weights[1]*nli + weights[2]*prec
                scored.append((ans, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            vote_pool = []
            for rank, (ans, _) in enumerate(scored, 1):
                if rank <= 5:
                    vote_pool.extend([ans] * rank_counts.get(rank, 0))
            final = Counter(vote_pool).most_common(1)
            updated_item["ccqa_answer"] = final[0][0] if final else top_answer
            updated_item["ccqa_method"] = "precision_rank_weighted_vote"
        else:
            updated_item["ccqa_answer"] = top_answer
            updated_item["ccqa_method"] = "majority_vote"
        updated_results.append(updated_item)
    return updated_results

def analyze_per_answer_count(results, method_field, correct_field):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in results:
        correct_answer = item.get(correct_field)
        all_answers = [extract_numerical_answer(item.get(f"response_{i}")) for i in range(1, 6)]
        match_count = sum(1 for a in all_answers if is_answer_correct(a, correct_answer))
        pred = item.get(method_field)
        if match_count > 0:
            stats[match_count]["total"] += 1
            if is_answer_correct(pred, correct_answer):
                stats[match_count]["correct"] += 1
    return stats

def visualize_accuracy_by_count(model_name, orig_stats, ccqa_stats, output_dir):
    keys = sorted(set(orig_stats.keys()) | set(ccqa_stats.keys()))
    orig_acc = []
    ccqa_acc = []
    labels = []
    for k in keys:
        labels.append(str(k))
        orig = orig_stats[k]
        ccqa = ccqa_stats[k]
        orig_acc.append(orig["correct"] / orig["total"] if orig["total"] else 0)
        ccqa_acc.append(ccqa["correct"] / ccqa["total"] if ccqa["total"] else 0)
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x, orig_acc, width=width, label="Original SC", color='orange')
    plt.bar([p + width for p in x], ccqa_acc, width=width, label="CCQA", color='skyblue')
    plt.xticks([p + width / 2 for p in x], labels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Matching Answers in 5")
    plt.title(f"{model_name} - Accuracy by Matching Answer Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_by_answer_count.png"))
    plt.close()

def process_file(file_path, weights):
    filename = os.path.basename(file_path)
    model = filename.split("_")[1]
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data
    results = apply_original_self_consistency(results)
    results = apply_new_self_consistency(results, weights=weights)
    correct_field = "answerKey" if "answerKey" in results[0] else "correct_answer"
    orig_correct = sum(is_answer_correct(item.get("original_sc_answer"), item.get(correct_field)) for item in results)
    ccqa_correct = sum(is_answer_correct(item.get("ccqa_answer"), item.get(correct_field)) for item in results)
    total = len(results)
    # 추가 분석 및 시각화
    orig_stats = analyze_per_answer_count(results, "original_sc_answer", correct_field)
    ccqa_stats = analyze_per_answer_count(results, "ccqa_answer", correct_field)
    visualize_accuracy_by_count(model, orig_stats, ccqa_stats, os.path.dirname(file_path))
    return {
        "model_name": model,
        "total_items": total,
        "original_sc_accuracy": orig_correct / total,
        "ccqa_accuracy": ccqa_correct / total,
        "original_sc_correct_count": orig_correct,
        "ccqa_correct_count": ccqa_correct,
        "weights": weights
    }

def create_comparison_csv(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ccqa_comparison.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "weight_cosine", "weight_entailment", "weight_precision",
            "original_sc_accuracy", "ccqa_accuracy", "improvement", "total_items"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model_name": r["model_name"],
                "weight_cosine": r["weights"][0],
                "weight_entailment": r["weights"][1],
                "weight_precision": r["weights"][2],
                "original_sc_accuracy": f"{r['original_sc_accuracy']:.4f}",
                "ccqa_accuracy": f"{r['ccqa_accuracy']:.4f}",
                "improvement": f"{r['ccqa_accuracy'] - r['original_sc_accuracy']:.4f}",
                "total_items": r["total_items"]
            })
    print(f"\n📄 CSV 저장 완료: {out_path}")

def visualize_comparison(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("improvement", ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df["improvement"], tick_label=[f"({c:.1f},{n:.1f},{p:.1f})" for c,n,p in zip(df.weight_cosine, df.weight_entailment, df.weight_precision)], color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy Improvement")
    plt.title("Top 10 Accuracy Improvements by Weight Combination")
    plt.tight_layout()
    plt.grid(True)
    output_dir = os.path.dirname(csv_path)
    plt.savefig(os.path.join(output_dir, "ccqa_comparison.png"), dpi=300)
    plt.show()

def main():
    input_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_precision_nli_scored"
    output_dir = os.path.join(input_dir, "ccqa_comparison")
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    steps = [round(0.1 * i, 2) for i in range(11)]
    weight_sets = [(c, n, p) for c, n, p in product(steps, repeat=3) if abs(c + n + p - 1.0) < 1e-6]
    all_results = []
    for weights in weight_sets:
        print(f"\n테스트 중: weights = cosine {weights[0]}, entail {weights[1]}, precision {weights[2]}")
        for file_path in tqdm(json_files, desc=f"W={weights}", leave=False):
            result = process_file(file_path, weights)
            all_results.append(result)
    create_comparison_csv(all_results, output_dir)
    visualize_comparison(os.path.join(output_dir, "ccqa_comparison.csv"))

if __name__ == "__main__":
    main()
