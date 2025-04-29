import os
import json
import glob
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import re

# ===== 경로 설정 =====
input_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate"
output_path = "wrong_vote_distribution.png"

# ===== 초기화 =====
vote_frequencies = Counter()

# ===== 정답 추출 패턴 통일 =====
patterns = [
    r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)',
    r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
    r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
    r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?'
]

def extract_answer(text):
    if not text:
        return None
    text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().upper()
    return None

# ===== JSON 순회 =====
files = glob.glob(os.path.join(input_dir, "*.json"))
for file_path in tqdm(files, desc="Analyzing files"):
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data

    for item in results:
        correct = item.get("answerKey") or item.get("correct_answer")
        responses = [item.get(f"response_{i}") for i in range(1, 6)]
        extracted_answers = [extract_answer(r) for r in responses if r]

        # 투표 진행
        vote_counter = Counter(extracted_answers)
        if not vote_counter:
            continue

        voted, voted_count = vote_counter.most_common(1)[0]

        if voted != correct:
            vote_frequencies[voted_count] += 1

# ===== 그래프 =====
x = sorted(vote_frequencies.keys())
y = [vote_frequencies[k] for k in x]

plt.figure(figsize=(8, 5))
plt.bar(x, y, color="salmon")
plt.xlabel("Vote Count of Chosen Answer (Wrong Cases)")
plt.ylabel("Number of Samples")
plt.title("Frequency of Votes for Incorrectly Chosen Answers (Self-Consistency)")
plt.xticks(x)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ 저장 완료: {output_path}")
