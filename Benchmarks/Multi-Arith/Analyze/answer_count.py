import os
import json
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd


def extract_numerical_answer(response):
    if not response:
        return None
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
    ]
    text = response.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().lower()
    return None


def is_answer_equal(a, b):
    return a is not None and b is not None and str(a).strip().lower() == str(b).strip().lower()


def analyze_by_answer_presence(results):
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    no_answer_stats = {"total": 0, "correct": 0}

    for item in results:
        if not isinstance(item, dict):
            continue

        correct_answer = item.get("correct_answer") or item.get("answerKey") or item.get("original_answer")
        if not correct_answer:
            continue

        extracted_answers = []
        for i in range(1, 6):
            response = item.get(f"response_{i}")
            if response:
                extracted = extract_numerical_answer(response)
                extracted_answers.append(extracted)

        # 유효한 답변만 필터링 (None이 아닌 것만)
        valid_answers = [ans for ans in extracted_answers if ans]
        
        # 유효한 답변이 없는 경우
        if not valid_answers:
            no_answer_stats["total"] += 1
            continue

        vote_counts = Counter(valid_answers)
        most_common_answer, _ = vote_counts.most_common(1)[0]
        is_correct = is_answer_equal(most_common_answer, correct_answer)

        correct_matches = sum(is_answer_equal(ans, correct_answer) for ans in valid_answers)

        # 정답이 있는 경우 통계 업데이트
        stats[correct_matches]["total"] += 1
        if is_correct:
            stats[correct_matches]["correct"] += 1
            
    return stats, no_answer_stats

def save_stats_to_csv(stats, no_answer_stats, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Correct Answer Count", "Total", "Correct", "Accuracy"])
        for k in sorted(stats):
            total = stats[k]["total"]
            correct = stats[k]["correct"]
            acc = correct / total if total > 0 else 0
            writer.writerow([k, total, correct, f"{acc:.4f}"])
        writer.writerow(["No Answer", no_answer_stats["total"], no_answer_stats["correct"],
                         f"{no_answer_stats['correct'] / no_answer_stats['total']:.4f}" if no_answer_stats["total"] else 0])
    print(f"📄 저장 완료: {output_csv}")

def plot_stats(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df["Correct Answer Count"] != "No Answer"]
    df["Correct Answer Count"] = df["Correct Answer Count"].astype(int)
    df = df.sort_values("Correct Answer Count")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Correct Answer Count"].astype(str), df["Accuracy"].astype(float), color='skyblue')
    plt.xlabel("Number of Correct Answers in 5 Responses")
    plt.ylabel("Self-Consistency Accuracy")
    plt.title("Accuracy by Number of Correct Responses")
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y')

    for bar, total, correct in zip(bars, df["Total"], df["Correct"]):
        height = bar.get_height()
        # 텍스트 위치를 막대 높이의 중간 정도로 조정
        # height/2는 막대 중간, height*0.7은 막대 높이의 70% 지점
        text_y_position = height * 0.7  # 이 값을 조정하여 텍스트 위치 변경
        plt.text(bar.get_x() + bar.get_width()/2, text_y_position, f"{correct}/{total}", 
                ha='center', fontsize=9, color='black')

    plt.tight_layout()
    out_path = csv_file.replace(".csv", ".png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"🖼 그래프 저장 완료: {out_path}")


def process_all_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    combined_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    combined_no_answer = {"total": 0, "correct": 0}

    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get("results", data)
        stats, no_answer = analyze_by_answer_presence(results)
        for k, v in stats.items():
            combined_stats[k]["total"] += v["total"]
            combined_stats[k]["correct"] += v["correct"]
        combined_no_answer["total"] += no_answer["total"]
        combined_no_answer["correct"] += no_answer["correct"]

        model_name = os.path.splitext(file)[0]
        output_csv = os.path.join(output_dir, f"{model_name}_answer_presence.csv")
        save_stats_to_csv(stats, no_answer, output_csv)
        plot_stats(output_csv)

    combined_csv = os.path.join(output_dir, "combined_answer_presence.csv")
    save_stats_to_csv(combined_stats, combined_no_answer, combined_csv)
    plot_stats(combined_csv)

if __name__ == "__main__":
    INPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/Multi-Arith/Result/ccqa_results"
    OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/Multi-Arith/Analyze/answer_count"
    process_all_files(INPUT_DIR, OUTPUT_DIR)
