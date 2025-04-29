"""
------------------------------
• GPT-4o mini API로 ‘답변→질문’ 데이터셋을 생성
• ThreadPoolExecutor로 병렬 호출 (동시 요청 수=workers)
• 429(속도 제한) 시 지수 백오프
• 테스트 모드(--test)·중간 저장 지원
"""

import json, requests, time, os, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------------------------
# 환경 설정
# --------------------------------------------------------------------
KEY_PATH            = Path(__file__).parent.parent / "openai_key.txt"
API_KEY             = KEY_PATH.read_text().strip()
API_URL             = "https://api.openai.com/v1/chat/completions"
MAX_OUTPUT_TOKENS   = 100          # 출력 토큰 상한
RETRY_LIMIT         = 10          # 429 재시도 횟수
CONCURRENCY_DEFAULT = 5          # 기본 동시 스레드 수
SAVE_INTERVAL_NORM  = 10           # 평시 저장 주기
SAVE_INTERVAL_TEST  = 1            # 테스트 모드 저장 주기
INPUT_PATH          = r"C:\Projects\CCQA_official\finetuning\qa_dataset.json"
OUTPUT_PATH         = r"C:\Projects\CCQA_official\finetuning\qa_dataset_gpt.json"
TEST_COUNT          = 3            # --test 모드 대상 개수

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------------------------
# Few-shot 예시 & 프롬프트 생성
# --------------------------------------------------------------------
math_examples = [
    {
        "response": ("Marco's dad's strawberries weighed 11 pounds. Together their strawberries "
                     "weighed 30 pounds. Marco's strawberries weigh 30 - 11 = 19 pounds. "
                     "The answer is 19."),
        "question": ("Marco and his dad went strawberry picking. Marco's dad's strawberries weighed "
                     "11 pounds. If together their strawberries weighed 30 pounds. How much did "
                     "Marco's strawberries weigh?")
    },
    {
        "response": ("There were 50 red macaroons and 40 green macaroons. If Fran ate 15 green "
                     "macaroons, she ate 15 * 2 = 30 red macaroons. Now she has 40 - 15 = 25 green "
                     "macaroons and 50 - 30 = 20 red macaroons. So Fran has 25 + 20 = 45 macaroons "
                     "left. The answer is 45."),
        "question": ("Fran baked 50 red macaroons and 40 green macaroons. How many macaroons will "
                     "remain if Fran ate 15 green macaroons and twice as many red macaroons as "
                     "green macaroons?")
    }
]

commonsense_examples = [
    {
        "response": ("Jewelry store, B: Neck, C: Jewlery box, D: Jewelry box, E: Boutique\n\n"
                     "The answer must be a location where one can purchase jewelry. Among the given "
                     "options, a jewelry store is specifically designed for this purpose. So the "
                     "answer is (A)."),
        "question": "To locate a choker not located in a jewelry box or boutique where would you go?"
    },
    {
        "response": ("The answer should be the effect of playing soccer for a long time. Of the above "
                     "choices, the best answer is E. So the answer is E."),
        "question": "What does playing soccer for a long time lead to?"
    }
]

strategyqa_examples = [
    {
        "response": ("Mixed martial arts is not totally original from Roman Colosseum games. While it "
                     "originated in ancient Rome, it evolved over time. It is a relatively modern "
                     "discipline. Thus, it is not a direct copy from Roman Colosseum games. So the "
                     "answer is no."),
        "question": "Is Mixed martial arts totally original from Roman Colosseum games?"
    },
    {
        "response": ("Hawaiian cuisine includes dishes such as poke, laulau, and kalua pig. These dishes "
                     "contain fish and pork. Thus, the cuisine of Hawaii is not suitable for a vegan. "
                     "So the answer is no."),
        "question": "Is the cuisine of Hawaii suitable for a vegan?"
    }
]

BASE_PROMPT = (
    "Please don't generate information that is not in the answer.\n"
    "Important:\n"
    "1. Generate ONLY the question itself, without any additional instructions or meta-commentary.\n"
    "2. Please generate a question using numbers or main words that exist in the answer."
)

def create_prompt(example: dict, dataset_type: str) -> str:
    """데이터셋 유형에 맞는 Few-shot 프롬프트 생성"""
    ds = dataset_type.lower()
    if ds in {"gsm8k", "svamp", "mawps"}:
        ex_text = "\n\n".join(
            f"Answer: {e['response']}\nQuestion: {e['question']}" for e in math_examples)
        return (
            "Generate the original question for the given math problem answer. "
            "Do not change any numeric values in the answer.\n"
            f"{BASE_PROMPT}\n\nExamples:\n{ex_text}\n"
            "Now, generate a question for this answer:\n"
            f"{example['response']}\nQuestion:"
        )

    if ds in {"commonsenseqa", "commonsense_qa"}:
        ex_text = "\n\n".join(
            f"Answer: {e['response']}\nQuestion: {e['question']}" for e in commonsense_examples)
        return (
            "From the commonsense reasoning answer provided below, recreate the original question. "
            "Do not include choices in your question.\n"
            f"{BASE_PROMPT}\n\nExamples:\n{ex_text}\n"
            "Now, generate a question for this answer:\n"
            f"{example['response']}\nQuestion:"
        )

    if ds in {"strategyqa", "strategy_qa"}:
        ex_text = "\n\n".join(
            f"Answer: {e['response']}\nQuestion: {e['question']}" for e in strategyqa_examples)
        return (
            "Create a yes/no question that would have this as its answer.\n"
            f"{BASE_PROMPT}\n\nExamples:\n{ex_text}\n"
            "Now, generate a question for this answer:\n"
            f"{example['response']}\nQuestion:"
        )

    # 기본
    return (
        f"{BASE_PROMPT}\n\nAnswer: {example['response']}\n"
        "Now, generate the original question.\nQuestion:"
    )

# --------------------------------------------------------------------
# OpenAI 호출 (429 백오프 포함)
# --------------------------------------------------------------------
def call_openai(prompt: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates questions based on answers."},
            {"role": "user",    "content": prompt}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
        "top_p": 0.9
    }

    for attempt in range(RETRY_LIMIT):
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if resp.status_code == 429:
            sleep_sec = 2 ** attempt
            print(f"[429] Rate-limit… {sleep_sec}s 후 재시도 (try {attempt+1}/{RETRY_LIMIT})")
            time.sleep(sleep_sec); continue
        resp.raise_for_status()
        time.sleep(2.5)  # 또는 3초
        txt = resp.json()["choices"][0]["message"]["content"].strip()
        return txt[1:-1].strip() if txt.startswith('"') and txt.endswith('"') else txt
    raise RuntimeError("재시도 한도 초과")

# --------------------------------------------------------------------
# 메인 로직
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GPT-4o mini 병렬 질문 생성기")
    p.add_argument("--test",    action="store_true", help="테스트 모드 (3개 항목만)")
    p.add_argument("--workers", type=int, default=CONCURRENCY_DEFAULT,
                   help=f"동시 요청 스레드 수 (기본 {CONCURRENCY_DEFAULT})")
    return p.parse_args()

def main():
    args       = parse_args()
    test_mode  = args.test
    workers    = args.workers
    out_path   = OUTPUT_PATH.replace(".json", "_test.json") if test_mode else OUTPUT_PATH
    save_every = SAVE_INTERVAL_TEST if test_mode else SAVE_INTERVAL_NORM

    with open(INPUT_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[+] 로드: {len(dataset)}개 항목")

    # 기존 처리 여부 확인
    processed_keys = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prior = json.load(f)
        for it in prior:
            if it.get("generated_question"):
                k = f"{it.get('model','')}-{it.get('dataset_type','')}-{it['response'][:50]}"
                processed_keys.add(k)
        dataset = prior
        print(f"[+] 기존 결과 반영: {len(processed_keys)}개 이미 처리")

    # 대상 필터
    items = [
        it for it in dataset
        if it.get("is_correct") is False
        and f"{it.get('model','')}-{it.get('dataset_type','')}-{it['response'][:50]}" not in processed_keys
    ]
    if test_mode:
        items = items[:TEST_COUNT]
    print(f"[+] 처리 대상: {len(items)}개 (workers={workers})")
    if not items: return

    # 병렬 실행
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for it in items:
            futures[pool.submit(call_openai, create_prompt(it, it.get('dataset_type','unknown')))] = it

        done = 0
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                q = fut.result()
            except Exception as e:
                q = f"API 오류: {e}"
                print(f"[!] {e}")
            it["generated_question"] = q
            it["question_generated"] = True
            done += 1
            print(f"[{done}/{len(items)}] {it.get('dataset_type','?')} → {q[:70]}")

            if done % save_every == 0 or done == len(items):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print("  ↪ 중간 저장 완료")

    print(f"[✓] 완료! 결과: {out_path}")

if __name__ == "__main__":
    main()
