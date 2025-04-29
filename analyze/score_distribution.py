import os
import json
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

sns.set(style="whitegrid")


def extract_answer(text):
    if not text:
        return None
    import re
    text = text.lower()
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def collect_scores(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.json'))
    correct_cosine, incorrect_cosine = [], []
    correct_entail, incorrect_entail = [], []
    correct_contrastive, incorrect_contrastive = [], []
    for file_path in tqdm(files, desc="Processing"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        results = data["results"] if isinstance(data, dict) else data

        for item in results:
            correct_answer = item.get("answerKey") or item.get("correct_answer")
            if not correct_answer:
                continue

            for i in range(1, 6):
                response = item.get(f"response_{i}")
                if not response:
                    continue
                ans = extract_answer(response)
                sim = item.get("similarity_scores", {}).get(f"question_{i}", {})
                cosine = sim.get("cosine", None)
                entail = sim.get("entailment_backward", None)
                contrastive = sim.get("contrastive", None)

                if cosine is None or entail is None:
                    continue

                if ans == correct_answer:
                    correct_cosine.append(cosine)
                    correct_entail.append(entail)
                    correct_contrastive.append(contrastive)
                else:
                    incorrect_cosine.append(cosine)
                    incorrect_entail.append(entail)
                    incorrect_contrastive.append(contrastive)

    return correct_cosine, incorrect_cosine, correct_entail, incorrect_entail, correct_contrastive, incorrect_contrastive

def collect_composite_scores(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.json'))
    correct_composite, incorrect_composite = [], []
    for file_path in tqdm(files, desc="Processing"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        results = data["results"] if isinstance(data, dict) else data

        for item in results:
            correct_answer = item.get("answerKey") or item.get("correct_answer")
            if not correct_answer:
                continue

            for i in range(1, 6):
                response = item.get(f"response_{i}")
                if not response:
                    continue
                ans = extract_answer(response)
                sim = item.get("similarity_scores", {}).get(f"question_{i}", {})
                cosine = sim.get("cosine", None)
                entail = sim.get("entailment_backward", None)
                contrastive = sim.get("contrastive", None)

                if cosine is None or entail is None or contrastive is None:
                    continue

                composite_score = cosine + entail + contrastive

                if ans == correct_answer:
                    correct_composite.append(composite_score)
                else:
                    incorrect_composite.append(composite_score)

    return correct_composite, incorrect_composite

def plot_distributions(correct, incorrect, label, output_dir="plots"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    sns.kdeplot(correct, label="Correct", fill=True, color="skyblue")
    sns.kdeplot(incorrect, label="Incorrect", fill=True, color="salmon")
    plt.title(f"{label} Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    # 저장
    filename = label.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()



# 경로 설정
paths = {
    "GSM8K": "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate",
    # "CommonsenseQA": "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_precision_nli_scored"
}

# # 실행
# for name, path in paths.items():
#     print(f"\n=== {name} ===")
#     c_cos, i_cos, c_ent, i_ent, c_con, i_con = collect_scores(path)
#     print(f"# Samples - Correct: {len(c_cos)}, Incorrect: {len(i_cos)}")
#     plot_distributions(c_cos, i_cos, f"{name} Cosine")
#     plot_distributions(c_ent, i_ent, f"{name} Entailment")
#     plot_distributions(c_con, i_con, f"{name} Contrastive")
    
# 실행
for name, path in paths.items():
    print(f"\n=== {name} ===")
    c_score, i_score = collect_composite_scores(path)
    print(f"# Samples - Correct: {len(c_score)}, Incorrect: {len(i_score)}")
    plot_distributions(c_score, i_score, f"{name} Composite")

