import os, re, json, csv, glob, sys
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ---------------------------
#  CONFIG
# ---------------------------
INPUT_FOLDERS = [
    "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_precision_nli_scored",
]
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/combined_sc_comparison"
WEIGHT_SCHEMES = [[5, 5, 4, 4, 3], [1, 1, 1, 1, 1], [5, 4, 3, 2, 1]]
MAX_RESP = 5  

# ---------------------------
#  CORE HELPERS
# ---------------------------
ANSWER_PATTERNS = [
    r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)',
    r'\b(?:so\s+)?the\s+(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-Ea-e])\)?',
    r'\banswer\s*:?\s*\(?([A-Ea-e])\)?',
]


def extract_answer(txt: str) -> Optional[str]:
    if not txt:
        return None
    low = txt.lower()
    for p in ANSWER_PATTERNS:
        m = re.search(p, low)
        if m:
            return m.group(1).strip().upper()
    return None


def majority_vote(ans: List[Optional[str]]) -> Optional[str]:
    cnt = Counter(a for a in ans if a)
    if not cnt:
        return None
    top, top_n = cnt.most_common(1)[0]
    ties = [a for a, c in cnt.items() if c == top_n]
    return ties[0]  # first on tie


def responses(item: Dict[str, Any]) -> List[str]:
    return [item.get(f"response_{i}") for i in range(1, MAX_RESP + 1) if item.get(f"response_{i}")]


# ---------------------------
#  SELF‑CONSISTENCY
# ---------------------------

def apply_sc(items: List[Dict[str, Any]]) -> None:
    for it in items:
        ex = [extract_answer(r) for r in responses(it)]
        it["self_consistency_extraction"] = ex
        it["self_consistency_answer"] = majority_vote(ex)


def apply_weighted_sc(items: List[Dict[str, Any]], weights: List[int]):
    tag = "_".join(map(str, weights))
    for it in items:
        # fallback if similarity info missing
        sim = it.get("similarity_scores", {})
        # rank responses by f1 (missing -> 0)
        ranks = sorted(
            [(i, sim.get(f"question_{i}", {}).get("precision", 0.0)) for i in range(1, MAX_RESP + 1)],
            key=lambda x: x[1], reverse=True,
        )[: len(weights)]
        votes = []
        ex_ans = []
        for (idx, _), w in zip(ranks, weights):
            ans = extract_answer(it.get(f"response_{idx}"))
            ex_ans.append(ans)
            votes.extend([ans] * w if ans else [])
        it[f"weighted_vote_extracted_{tag}"] = ex_ans
        it[f"weighted_sc_answer_{tag}"] = majority_vote(votes) if votes else None


def correct(pred: Optional[str], gold: Optional[str]) -> bool:
    return bool(pred and gold and pred.upper() == gold.upper())

# ---------------------------
#  FILE PROCESSING
# ---------------------------

def model_from_name(fn: str) -> str:
    for pat in [r"CommonsenseQA_([^_]+)_", r"CSQA_([^_]+)_", r"([^_]+)_CommonsenseQA", r"([^_]+)_CSQA"]:
        m = re.search(pat, fn)
        if m:
            return m.group(1)
    return fn.replace(".json", "")


def sim_method(folder: str) -> str:
    name = os.path.basename(folder.rstrip("/"))
    return name.replace("_similar", "")


def process(path: str) -> Dict[str, Any]:
    data = json.load(open(path))
    res = data.get("results", data)
    for it in res:
        it["correct_answer"] = it.get("answerKey") or it.get("correct_answer")
    apply_sc(res)
    for ws in WEIGHT_SCHEMES:
        apply_weighted_sc(res, ws)
    total = len(res)
    summary = {
        "cot_correct": sum(correct(extract_answer(it.get("response_1")), it["correct_answer"]) for it in res),
        "sc_correct": sum(correct(it["self_consistency_answer"], it["correct_answer"]) for it in res),
        "weighted": defaultdict(int),
    }
    for ws in WEIGHT_SCHEMES:
        tag = "_".join(map(str, ws))
        summary["weighted"][tag] = sum(correct(it.get(f"weighted_sc_answer_{tag}"), it["correct_answer"]) for it in res)
    return {
        "model": model_from_name(os.path.basename(path)),
        "cot_acc": summary["cot_correct"] / total,
        "sc_acc": summary["sc_correct"] / total,
        "weighted": {k: v / total for k, v in summary["weighted"].items()},
        "total": total,
    }

# ---------------------------
#  CSV UTILS
# ---------------------------

def write_csv(all_rows: List[Dict[str, Any]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    main_csv = os.path.join(out_dir, "multi_folder_sc_comparison.csv")
    headers = ["similarity_method", "model_name", "cot_accuracy", "sc_accuracy"] + [
        f"weighted_sc_{'_'.join(map(str, ws))}" for ws in WEIGHT_SCHEMES
    ] + ["total_items"]
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"Saved summary to {main_csv}")

# ---------------------------
#  MAIN
# ---------------------------

def main():
    out = []
    if len(sys.argv) > 1:
        global OUTPUT_DIR
        OUTPUT_DIR = sys.argv[1]
    for folder in INPUT_FOLDERS:
        sim = sim_method(folder)
        files = glob.glob(os.path.join(folder, "*.json"))
        print(f"[{sim}] {len(files)} files")
        for fp in tqdm(files, desc=sim):
            res = process(fp)
            row = {
                "similarity_method": sim,
                "model_name": res["model"],
                "cot_accuracy": f"{res['cot_acc']:.4f}",
                "sc_accuracy": f"{res['sc_acc']:.4f}",
                "total_items": res["total"],
            }
            for ws in WEIGHT_SCHEMES:
                tag = "_".join(map(str, ws))
                row[f"weighted_sc_{tag}"] = f"{res['weighted'][tag]:.4f}"
            out.append(row)
    write_csv(out, OUTPUT_DIR)

if __name__ == "__main__":
    main()
