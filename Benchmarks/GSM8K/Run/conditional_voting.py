import os
import json
import re
import glob
from collections import Counter
from tqdm import tqdm
import csv
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

def extract_numerical_answer(response):
    if not response:
        return None
    patterns = [
    r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)',
    r'the (?:correct )?answer is\s*:?\s*\(?([A-E])\)?',
    r'(?:correct )?answer is\s*:?\s*\(?([A-Ea-e])\)?',
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
    """
    간단 비교용:
    - 만약 숫자로 해석 가능하면 float 변환 뒤, 절대값 오차 < 1e-5 검사
    - 그렇지 않으면 단순 문자열 비교
    """
    if extracted_answer is None or correct_answer is None:
        return False
    extracted_numeric = re.sub(r'[^\d.]', '', str(extracted_answer).strip())
    correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
    try:
        if extracted_numeric and correct_numeric:
            return abs(float(extracted_numeric) - float(correct_numeric)) < 1e-5
        else:
            return str(extracted_answer).strip() == str(correct_answer).strip()
    except ValueError:
        return str(extracted_answer).strip() == str(correct_answer).strip()

def apply_original_self_consistency(results):
    """
    기존 self-consistency: 가장 많이 나온 답변 = 정답
    """
    updated_results = []
    for item in tqdm(results, desc="original self-consistency"):
        updated_item = item.copy()
        responses = [item.get(f"response_{i}") for i in range(1, 6) if item.get(f"response_{i}")]
        if not responses:
            updated_item["original_sc_answer"] = None
            updated_results.append(updated_item)
            continue
        extracted_answers = [extract_numerical_answer(r) for r in responses]
        counts = Counter(a for a in extracted_answers if a)
        top = counts.most_common(1)
        updated_item["original_sc_answer"] = top[0][0] if top else None
        updated_results.append(updated_item)
    return updated_results

def apply_conditional_unified_voting(results, w_f=0.0, w_s=1.0, alpha=0.5, beta=0.5):
    """
    조건부 통합 방식:
    1) 먼저 최대 득표를 구한다. 만약 해당 득표 >= 3이면 그냥 majority 그대로.
    2) 아니면 (득표가 2 이하라면) -> 각 선택지별로 score = w_f * freq(A) + w_s * sum_of(RelScore_i)
       * RelScore_i = alpha*cosine + beta*entail
       * freq(A) = A를 골라준 응답 수
       * sum_of(RelScore_i) = A를 골라준 각각의 응답에 대해 합산
    3) 최고 score의 선택지를 최종 정답
    """
    updated = []
    for item in tqdm(results, desc="conditional unified voting"):
        new_item = item.copy()

        # 1. 응답/선택지 추출
        responses = [(i, extract_numerical_answer(item.get(f"response_{i}")))
                     for i in range(1, 6) if item.get(f"response_{i}")]
        # 응답 중에서 None 아닌 것만 추출
        valid = [(idx,a) for (idx,a) in responses if a]
        if not valid:
            new_item["ccqa_answer"] = None
            new_item["ccqa_method"] = "no_valid_response"
            updated.append(new_item)
            continue

        # 2. 득표 계산
        c = Counter(ans for (_,ans) in valid)
        top_choice, top_count = c.most_common(1)[0]

        # 3. 조건부 판단
        if top_count >= 3:
            # 과반수면 그대로
            new_item["ccqa_answer"] = top_choice
            new_item["ccqa_method"] = "majority_vote"
        else:
            # 빈도+유사도 통합으로 전체 스코어 계산
            all_choices = list(c.keys())
            scores_dict = {}
            for choice in all_choices:
                scores_dict[choice] = {"freq": c[choice], "rel_sum": 0.0}

            # 각 응답에 대해 RelScore계산 후, 해당 choice에 누적
            for idx, ans in valid:
                sim = item.get("similarity_scores", {}).get(f"question_{idx}", {})
                cos = sim.get("cosine", 0.0)
                ent = sim.get("entailment_backward", 0.0)
                # alpha*cos + beta*ent
                rel_score = alpha*cos + beta*ent
                scores_dict[ans]["rel_sum"] += rel_score

            # 최종 score(A) = w_f * freq(A) + w_s * (rel_sum)
            final_scores = {}
            for choice in all_choices:
                freq_part = w_f * scores_dict[choice]["freq"]
                rel_part  = w_s * scores_dict[choice]["rel_sum"]
                final_scores[choice] = freq_part + rel_part

            best_choice = max(final_scores, key=final_scores.get)
            new_item["ccqa_answer"] = best_choice
            new_item["ccqa_method"] = "freq+similarity_unified"
        updated.append(new_item)
    return updated

def process_file(file_path, w_f=0.2, w_s=1.0, alpha=0.5, beta=0.5):
    filename = os.path.basename(file_path)
    model = filename.replace(".json","")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # results 필드가 있으면 거기서, 없으면 data 자체
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data

    # 1) 기존 self-consistency
    results = apply_original_self_consistency(results)

    # 2) 조건부 통합
    results = apply_conditional_unified_voting(results,
                                              w_f=w_f,
                                              w_s=w_s,
                                              alpha=alpha,
                                              beta=beta)

    total = len(results)
    orig_correct, ccqa_correct = 0,0
    for it in results:
        c_ans = it.get("correct_answer")
        orig = it.get("original_sc_answer")
        ccqa = it.get("ccqa_answer")
        if is_answer_correct(orig, c_ans):
            orig_correct += 1
        if is_answer_correct(ccqa, c_ans):
            ccqa_correct += 1

    return {
        "model_name": model,
        "total_items": total,
        "orig_acc": orig_correct / total if total>0 else 0,
        "ccqa_acc": ccqa_correct / total if total>0 else 0,
        "orig_correct": orig_correct,
        "ccqa_correct": ccqa_correct
    }

def create_comparison_csv(all_results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name","w_f","w_s","alpha","beta",
            "orig_acc","ccqa_acc","improvement","total_items"
        ])
        writer.writeheader()
        for row in all_results:
            imp = row["ccqa_acc"] - row["orig_acc"]
            writer.writerow({
                "model_name": row["model_name"],
                "w_f": row["w_f"],
                "w_s": row["w_s"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "orig_acc": f"{row['orig_acc']:.4f}",
                "ccqa_acc": f"{row['ccqa_acc']:.4f}",
                "improvement": f"{imp:.4f}",
                "total_items": row["total_items"]
            })
    print(f"CSV 저장 완료: {out_path}")

def main():
    """
    - GSM8K / CSQA 파일을 대상으로
    - w_f, w_s, alpha, beta 등의 파라미터 grid를 돌려볼 수 있음
    - 예: w_f in {0.1, 0.2}, w_s in {1.0}, alpha/beta in {0.5, 0.5}
    """

    input_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate"  # 원하는 폴더
    output_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_t5_result/precision_nli_separate/conditional_ccqa"
    json_files = glob.glob(os.path.join(input_dir,"*.json"))

    # 간단히 예시
    w_f_candidates = [0.2]
    w_s_candidates = [1.0]
    alpha_beta_list = [(0.9,0.1)]

    all_results = []
    for w_f in w_f_candidates:
        for w_s in w_s_candidates:
            for (alpha, beta) in alpha_beta_list:
                for file_path in json_files:
                    res = process_file(file_path, w_f=w_f, w_s=w_s, alpha=alpha, beta=beta)
                    # 필요한 필드들 저장
                    summary = {
                        "model_name": res["model_name"],
                        "w_f": w_f,
                        "w_s": w_s,
                        "alpha": alpha,
                        "beta": beta,
                        "orig_acc": res["orig_acc"],
                        "ccqa_acc": res["ccqa_acc"],
                        "total_items": res["total_items"]
                    }
                    all_results.append(summary)

    out_csv = os.path.join(output_dir, "conditional_unified_comparison.csv")
    create_comparison_csv(all_results, out_csv)

if __name__ == "__main__":
    main()
