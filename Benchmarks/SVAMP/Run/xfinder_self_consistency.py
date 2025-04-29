import os
import json
import sys
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append('/data3/jykim/Projects/CCQA_official')

class XFinderExtractor:
    """xFinder 모델을 사용한 답변 추출기"""
    
    def __init__(self, model_name="IAAR-Shanghai/xFinder-qwen1505"):
        """
        xFinder 모델 초기화
        Args:
            model_name: 사용할 xFinder 모델 이름
        """
        print(f"xFinder 모델 '{model_name}' 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = self.model.device
        print(f"xFinder 모델 로딩 완료. 장치: {self.device}")
    
    def extract_answer(self, question: str, llm_response: str, key_answer_type: str, answer_range: str) -> str:
        """
        xFinder 모델을 사용하여 LLM 응답에서 키 답변 추출
        
        Args:
            question: 원본 질문
            llm_response: LLM 응답 텍스트
            key_answer_type: 키 답변 유형 (alphabet option, short text, categorical label, math)
            answer_range: 가능한 답변 범위
            
        Returns:
            추출된 답변 또는 [No valid answer]
        """
        # xFinder 프롬프트 형식으로 구성
        prompt = f"""You are a help assistant tasked with extracting the precise key answer from given output sentences. You must only provide the extracted key answer without including any additional text.
—
I will provide you with a question, output sentences along with an answer range. The output sentences are the response of the question provided. The answer range could either describe the type of answer expected or list all possible valid answers. Using the information provided, you must accurately and precisely determine and extract the intended key answer from the output sentences. Please don't have your subjective thoughts about the question. First, you need to determine whether the content of the output sentences is relevant to the given question. If the entire output sentences are unrelated to the question (meaning the output sentences are not addressing the question), then output [No valid answer]. Otherwise, ignore the parts of the output sentences that have no relevance to the question and then extract the key answer that matches the answer range. Below are some special cases you need to be aware of:
(1) If the output sentences present multiple different answers, carefully determine if the later provided answer is a correction or modification of a previous one. If so, extract this corrected or modified answer as the final response. Conversely, if the output sentences fluctuate between multiple answers without a clear final answer, you should output [No valid answer].
(2) If the answer range is a list and the key answer in the output sentences is not explicitly listed among the candidate options in the answer range, also output [No valid answer].
—
Question: {question}
Output sentences: {llm_response}
Key answer type: {key_answer_type}
Answer range: {answer_range}
Key extracted answer:"""

        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 생성 매개변수 설정
        generation_config = {
            "max_new_tokens": 32,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 5,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 답변 생성
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_config)
        
        # 응답 디코딩 및 후처리
        response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 응답 정리 (앞뒤 공백 제거)
        answer = response.strip()
        
        return answer

def load_benchmark_results(results_path: str) -> List[Dict[str, Any]]:
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        print(f"결과 로드 오류 {results_path}: {e}")
        return []

def apply_self_consistency(
    results: List[Dict[str, Any]],
    output_path: str,
    xfinder: XFinderExtractor
) -> List[Dict[str, Any]]:

    start_time = time.time()
    sc_results = []
    
    for i, item in enumerate(tqdm(results, desc="Self-consistency with xFinder 적용 중")):
        sc_item = item.copy()
        
        # 모든 응답 저장
        responses = []
        for j in range(1, 6):  # 5개 응답 처리
            response_key = f"response_{j}"
            if response_key in item and item[response_key]:
                responses.append(item[response_key])
        
        # 1. xFinder로 각 응답에서 답변 추출하여 voting
        answers_dict = {}
        original_question = item.get("body", "") + item.get("question", "")
        key_answer_type = "numerical"  # SVAMP 문제는 수학 문제이므로 numerical 타입
        answer_range = "a number"
        
        extracted_answers = []
        response_to_extracted_answer = {}  # 응답 텍스트와 추출된 정답 매핑
        
        for j, response in enumerate(responses):
            # xFinder로 답변 추출
            extracted_answer = xfinder.extract_answer(
                original_question,
                response,
                key_answer_type,
                answer_range
            )
            
            # 추출된 답변 저장
            extracted_answers.append(extracted_answer)
            sc_item[f"xfinder_answer_{j+1}"] = extracted_answer
            
            # 유효한 답변인 경우만 투표에 포함
            if extracted_answer and extracted_answer != "[No valid answer]":
                response_to_extracted_answer[response] = extracted_answer
                
                if extracted_answer in answers_dict:
                    answers_dict[extracted_answer] += 1
                else:
                    answers_dict[extracted_answer] = 1
        
        # 2. 가장 많이 나온 답변과 해당 응답 찾기
        if answers_dict:
            most_common_answer = max(answers_dict.items(), key=lambda x: x[1])
            most_common_value = most_common_answer[0]
            
            # most_common_value를 가진 첫 번째 응답을 majority_response로 저장
            majority_response = None
            for response, answer in response_to_extracted_answer.items():
                if answer == most_common_value:
                    majority_response = response
                    break
            
            # 저장
            sc_item["xfinder_most_common_answer"] = most_common_value
            sc_item["xfinder_vote_count"] = most_common_answer[1]
            sc_item["majority_response"] = majority_response
            sc_item["self_consistency_answer"] = most_common_value
            sc_item["total_valid_answers"] = sum(answers_dict.values())
            sc_item["all_xfinder_answers"] = answers_dict
        else:
            # xFinder로 어떤 유효한 답변도 찾지 못한 경우
            sc_item["xfinder_most_common_answer"] = None
            sc_item["xfinder_vote_count"] = 0
            sc_item["majority_response"] = None
            sc_item["self_consistency_answer"] = None
            sc_item["total_valid_answers"] = 0
            sc_item["all_xfinder_answers"] = {}
        
        sc_item["extracted_answers"] = extracted_answers
        sc_results.append(sc_item)
        
        # 중간 결과 저장
        if (i + 1) % 10 == 0 or i == 0 or i == len(results) - 1:
            # 진행 상황 포함하여 결과 저장
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            result_with_progress = {
                "completed_questions": i + 1,
                "total_questions": len(results),
                "elapsed_time_seconds": elapsed_time,
                "elapsed_time_minutes": elapsed_time / 60,
                "results": sc_results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_with_progress, f, ensure_ascii=False, indent=2)
            
            # 진행 상황 출력
            print(f"진행 상황: {i + 1}/{len(results)} 문제 완료 ({((i + 1) / len(results) * 100):.1f}%)")
    
    return sc_results

def run_self_consistency_with_xfinder(
    model_name: str,
    results_path: str,
    output_path: str
) -> Optional[Dict]:
    """
    특정 모델에 대해 xFinder를 이용한 self-consistency 실행
    
    Args:
        model_name: 모델 이름
        results_path: 결과 JSON 파일 경로
        output_path: 출력 파일 경로
        
    Returns:
        저장된 self-consistency 결과 정보 또는 실패 시 None
    """
    try:
        start_time = time.time()
        
        # 결과 로드
        results = load_benchmark_results(results_path)
        
        if not results:
            print(f"모델 {model_name}에 대한 결과를 찾을 수 없음")
            return None
        
        print(f"모델 {model_name}에 대한 {len(results)}개 결과 로드됨")
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # xFinder 초기화
        try:
            xfinder = XFinderExtractor()
        except Exception as xe:
            print(f"xFinder 초기화 오류: {xe}. xFinder 없이는 계속할 수 없습니다.")
            return None
        
        # Self-consistency 적용
        sc_results = apply_self_consistency(
            results=results,
            output_path=output_path,
            xfinder=xfinder
        )
        
        # 시간 계산
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 처리 시간 정보 추가
        time_info = {
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "processing_time_hours": processing_time / 3600,
            "questions_count": len(results),
            "avg_time_per_question": processing_time / len(results) if len(results) > 0 else 0
        }
        
        # 최종 결과와 시간 정보 저장
        result_with_time = {
            "time_info": time_info,
            "results": sc_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_time, f, ensure_ascii=False, indent=2)
        
        print(f"Self-consistency with xFinder 결과가 {output_path}에 저장됨")
        print(f"처리 시간: {processing_time:.2f}초 ({processing_time/60:.2f}분)")
        
        return {
            "path": output_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"모델 {model_name}에 대한 self-consistency with xFinder 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_models_self_consistency(
    results_dir: str,
    output_dir: str,
    benchmark_name: str,
    parallel: bool = False
) -> Dict[str, Dict]:
    """
    결과가 있는 모든 모델에 대해 self-consistency 실행
    
    Args:
        results_dir: 결과 JSON 파일이 포함된 디렉토리
        output_dir: 결과를 저장할 디렉토리
        benchmark_name: 벤치마크 이름 (예: 'svamp')
        parallel: 모델을 병렬로 실행할지 여부
        
    Returns:
        모델 이름을 출력 파일 경로와 시간 정보에 매핑하는 사전
    """
    # 디렉토리에서 모든 결과 파일 찾기
    result_files = [f for f in os.listdir(results_dir) if f.startswith(f"{benchmark_name}_") and f.endswith(".json")]
    model_names = [f.replace(f"{benchmark_name}_", "").replace(".json", "") for f in result_files]
    
    results = {}
    
    if parallel and len(model_names) > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 각 모델에 대한 작업 제출
            future_to_model = {}
            for model_name in model_names:
                results_path = os.path.join(results_dir, f"{benchmark_name}_{model_name}.json")
                output_path = os.path.join(output_dir, f"{benchmark_name}_{model_name}_self_consistency_xfinder_voting.json")
                
                future = executor.submit(
                    run_self_consistency_with_xfinder,
                    model_name,
                    results_path,
                    output_path
                )
                future_to_model[future] = model_name
            
            # 완료되는 대로 결과 수집
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_info = future.result()
                    results[model_name] = result_info
                    if result_info:
                        time_info = result_info["time_info"]
                        print(f"모델 {model_name}에 대한 self-consistency with xFinder voting 완료 (처리 시간: {time_info['processing_time_minutes']:.2f}분)")
                    else:
                        print(f"모델 {model_name}에 대한 self-consistency with xFinder voting 실패")
                except Exception as e:
                    print(f"모델 {model_name}에 대한 self-consistency with xFinder voting 실행 오류: {e}")
                    results[model_name] = None
    else:
        # 순차적으로 실행
        for model_name in model_names:
            try:
                results_path = os.path.join(results_dir, f"{benchmark_name}_{model_name}.json")
                output_path = os.path.join(output_dir, f"{benchmark_name}_{model_name}_self_consistency_xfinder_voting.json")
                
                result_info = run_self_consistency_with_xfinder(
                    model_name=model_name,
                    results_path=results_path,
                    output_path=output_path
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"모델 {model_name}에 대한 self-consistency with xFinder voting 실행 오류: {e}")
                results[model_name] = None
    
    return results

def run_svamp_self_consistency_xfinder_voting(
    results_dir: str,
    output_dir: str,
    parallel: bool = False
) -> Dict[str, Dict]:

    benchmark_name = "svamp"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"SVAMP 벤치마크에 대한 self-consistency with xFinder voting 시작...")
    results = run_all_models_self_consistency(
        results_dir=results_dir,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
        parallel=parallel
    )
    
    # 결과 요약 보고
    print("\nSelf-consistency with xFinder voting 완료. 결과 요약:")
    
    # 모든 모델 처리 시간 정보를 하나의 JSON 파일로 저장
    time_summary = {}
    
    for model_name, result_info in results.items():
        if result_info:
            time_info = result_info["time_info"]
            path = result_info["path"]
            time_summary[model_name] = time_info
            print(f"{model_name}:")
            print(f"   - 처리 시간: {time_info['processing_time_seconds']:.2f}초 ({time_info['processing_time_minutes']:.2f}분)")
            print(f"   - 평균 문제당 시간: {time_info['avg_time_per_question']:.2f}초")
            print(f"   - 출력 경로: {path}")
        else:
            print(f" {model_name}: 실패")
    
    # 시간 정보 요약 저장
    time_summary_path = os.path.join(output_dir, "self_consistency_xfinder_voting_time_summary.json")
    with open(time_summary_path, 'w', encoding='utf-8') as f:
        json.dump(time_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")
    print(f"시간 요약 정보는 {time_summary_path}에 저장되었습니다.")
    
    return results

# 설정 파라미터
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/xfinder_self_consistency"
RUN_PARALLEL = False  # 모델을 병렬로 실행하려면 True로 설정

# 메인 실행
if __name__ == "__main__":
    run_svamp_self_consistency_xfinder_voting(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        parallel=RUN_PARALLEL
    )