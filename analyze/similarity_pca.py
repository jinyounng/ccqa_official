import os
import json
import glob
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 정답 추출 함수
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
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().upper()
    return None

# JSON에서 점수와 정답 여부 수집
def collect_data(folder_path):
    X = []  # features: [cosine, entailment, contrastive]
    y = []  # labels: 1 (correct), 0 (incorrect)
    
    files = glob.glob(os.path.join(folder_path, "*.json"))
    for file_path in tqdm(files, desc="Collecting"):
        with open(file_path, "r") as f:
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
                cosine = sim.get("cosine")
                entail = sim.get("entailment_backward")
                contrastive = sim.get("contrastive")
                if None in (cosine, entail, contrastive):
                    continue
                X.append([cosine, entail, contrastive])
                y.append(1 if ans == correct_answer else 0)
    return np.array(X), np.array(y)

# PCA 시각화
def plot_pca(X, y, title="PCA of Similarity Scores"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Incorrect", alpha=0.5, c="salmon")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Correct", alpha=0.5, c="skyblue")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_similarity_scores.png", dpi=300)
    plt.show()

# 실행
if __name__ == "__main__":
    folder = "/data3/jykim/Projects/CCQA_official/Benchmarks/CommonsenseQA/Results/ccqa_result/t5_0_results/cosine_precision_nli_scored"
    X, y = collect_data(folder)
    print(f"Total samples: {len(X)}")
    plot_pca(X, y, title="CommonsenseQA PCA (Cosine + Entailment + Contrastive)")
