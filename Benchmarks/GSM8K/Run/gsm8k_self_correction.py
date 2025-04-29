import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

def load_gsm8k_results(results_path: str) -> List[Dict[str, Any]]:
    """
    GSM8K 결과 JSON 파일 로드
    
    Args:
        results_path: 결과 JSON 파일 경로
        
    Returns:
        결과 항목 리스트
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"결과 로드 오류 {results_path}: {e}")
        return []

def apply_self_refinement(
    runner: LLMRunner,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> List[Dict[str, Any]]:
    """
    결과에 self-refinement 적용하고 문제 하나마다 결과 저장
    
    Args:
        runner: LLMRunner 인스턴스
        results: 결과 항목 리스트
        output_path: 결과를 저장할 파일 경로
        max_new_tokens: 최대 생성 토큰 수
        temperature: 생성 temperature
        top_p: top-p 샘플링 파라미터
        
    Returns:
        self-refinement 응답이 포함된 결과 항목 리스트
    """
    refined_results = []
    
    for i, item in enumerate(tqdm(results, desc="Self-refinement 적용 중")):
        refined_item = item.copy()
        
        # 첫 번째 응답 가져오기
        first_response = item.get("response_1", "")
        
        if not first_response:
            print(f"경고: 첫 번째 응답을 찾을 수 없습니다. 질문: {item.get('question', 'unknown')}")
            refined_results.append(refined_item)
            
            # 현재까지의 결과 저장
            current_results = {
                "completed_questions": i + 1,
                "total_questions": len(results),
                "results": refined_results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
                
            continue
        
        # 리뷰 프롬프트 생성
        review_prompt = f"""Q: {item.get('question', '')}

Your answer was:
{first_response}

Review your previous answer and find problems with your answer."""
        
        try:
            # 리뷰 응답 생성
            review_responses = runner.generate_responses(
                prompt=review_prompt,
                num_responses=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=False
            )
            
            review_response = review_responses[0] if review_responses else ""
            refined_item["first_refine"] = review_response
            
            # 개선 프롬프트 생성
            improvement_prompt = f"""Q: {item.get('question', '')}

Your answer was:
{first_response}

Review of your answer:
{review_response}

Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form /the answer is {{answer}}."""
            
            # 개선 응답 생성
            improvement_responses = runner.generate_responses(
                prompt=improvement_prompt,
                num_responses=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=False
            )
            
            improvement_response = improvement_responses[0] if improvement_responses else ""
            refined_item["second_refine"] = improvement_response
            
        except Exception as e:
            print(f"Self-refinement 적용 오류: {e}")
            refined_item["first_refine"] = f"Error: {str(e)}"
            refined_item["second_refine"] = f"Error: {str(e)}"
        
        refined_results.append(refined_item)
        
        # 문제 하나 처리할 때마다 결과 저장
        current_results = {
            "completed_questions": i + 1,
            "total_questions": len(results),
            "results": refined_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        
        # 진행 상황 출력
        if (i + 1) % 10 == 0 or i == 0 or i == len(results) - 1:
            print(f"진행 상황: {i + 1}/{len(results)} 문제 완료 ({((i + 1) / len(results) * 100):.1f}%)")
    
    return refined_results

def run_self_refinement_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> Optional[Dict]:
    """
    특정 모델에 대해 self-refinement 실행
    
    Args:
        model_name: 사용할 모델 이름
        results_dir: 결과 JSON 파일이 포함된 디렉토리
        output_dir: 개선된 결과를 저장할 디렉토리
        max_new_tokens: 최대 생성 토큰 수
        temperature: 생성 temperature
        top_p: top-p 샘플링 파라미터
        
    Returns:
        저장된 개선 결과 정보 (경로 및 처리 시간) 또는 실패 시 None
    """
    try:
        start_time = time.time()
        
        # 이 모델의 결과 파일 찾기
        results_filename = f"gsm8k_{model_name}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        if not os.path.exists(results_path):
            print(f"모델 {model_name}에 대한 결과 파일을 찾을 수 없음: {results_path}")
            return None
        
        # 결과 로드
        results_data = load_gsm8k_results(results_path)
        
        # 결과 구조 확인 (시간 정보가 있는 새 형식 vs 없는 옛 형식)
        if isinstance(results_data, dict) and "results" in results_data:
            results = results_data["results"]
        else:
            results = results_data
            
        if not results:
            print(f"{results_path}에서 결과를 찾을 수 없음")
            return None
        
        print(f"모델 {model_name}에 대한 {len(results)}개 결과 로드됨")
        
        # 출력 디렉토리가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 개선된 결과 저장 경로
        refined_filename = f"gsm8k_{model_name}_refined.json"
        refined_path = os.path.join(output_dir, refined_filename)
        
        # 모델 초기화
        runner = LLMRunner(model_name)
        
        # Self-refinement 적용 (문제 하나마다 결과 저장)
        refined_results = apply_self_refinement(
            runner=runner,
            results=results,
            output_path=refined_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
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
            "results": refined_results
        }
        
        with open(refined_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_time, f, ensure_ascii=False, indent=2)
        
        print(f"개선된 결과가 {refined_path}에 저장됨")
        print(f"처리 시간: {processing_time:.2f}초 ({processing_time/60:.2f}분)")
        
        return {
            "path": refined_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"모델 {model_name}에 대한 self-refinement 실행 오류: {e}")
        return None

def run_all_models_refinement(
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9,
    parallel: bool = False
) -> Dict[str, Dict]:
    """
    결과가 있는 모든 모델에 대해 self-refinement 실행
    
    Args:
        results_dir: 결과 JSON 파일이 포함된 디렉토리
        output_dir: 개선된 결과를 저장할 디렉토리
        max_new_tokens: 최대 생성 토큰 수
        temperature: 생성 temperature
        top_p: top-p 샘플링 파라미터
        parallel: 모델을 병렬로 실행할지 여부
        
    Returns:
        모델 이름을 출력 파일 경로와 시간 정보에 매핑하는 사전
    """
    # 디렉토리에서 모든 결과 파일 찾기
    result_files = [f for f in os.listdir(results_dir) if f.startswith("gsm8k_") and f.endswith(".json")]
    model_names = [f.replace("gsm8k_", "").replace(".json", "") for f in result_files]
    
    results = {}
    
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 각 모델에 대한 작업 제출
            future_to_model = {}
            for model_name in model_names:
                future = executor.submit(
                    run_self_refinement_for_model,
                    model_name,
                    results_dir,
                    output_dir,
                    max_new_tokens,
                    temperature,
                    top_p
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
                        print(f"모델 {model_name}에 대한 self-refinement 완료 (처리 시간: {time_info['processing_time_minutes']:.2f}분)")
                    else:
                        print(f"모델 {model_name}에 대한 self-refinement 실패")
                except Exception as e:
                    print(f"모델 {model_name}에 대한 self-refinement 실행 오류: {e}")
                    results[model_name] = None
    else:
        # 순차적으로 실행
        for model_name in model_names:
            try:
                result_info = run_self_refinement_for_model(
                    model_name=model_name,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"모델 {model_name}에 대한 self-refinement 실행 오류: {e}")
                results[model_name] = None
    
    return results

# 설정 파라미터
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/gsm8k_results"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/self_correction_result"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.5
TOP_P = 0.9
RUN_PARALLEL = False  # 모델을 병렬로 실행하려면 True로 설정

# 메인 실행
if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모델을 두 그룹으로 나누기
    model_groups = [
        ["qwen-0.5b", "qwen-1.5b", "qwen-3b"],  # 첫 번째 그룹
        ["gemma-1b", "llama-1b", "llama-3b"]    # 두 번째 그룹
    ]
    
    all_results = {}
    
    # 각 그룹을 순차적으로 실행
    for group_idx, model_group in enumerate(model_groups):
        print(f"\n=== 실행 그룹 {group_idx+1}/{len(model_groups)}: {model_group} ===\n")
        
        group_results = {}
        
        if RUN_PARALLEL:
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                # 그룹 내 각 모델에 대한 작업 제출
                future_to_model = {}
                for model_name in model_group:
                    # 이 모델의 결과 파일 찾기
                    results_filename = f"gsm8k_{model_name}.json"
                    results_path = os.path.join(RESULTS_DIR, results_filename)
                    
                    if not os.path.exists(results_path):
                        print(f"모델 {model_name}에 대한 결과 파일을 찾을 수 없음: {results_path}")
                        continue
                        
                    future = executor.submit(
                        run_self_refinement_for_model,
                        model_name,
                        RESULTS_DIR,
                        OUTPUT_DIR,
                        MAX_NEW_TOKENS,
                        TEMPERATURE,
                        TOP_P
                    )
                    future_to_model[future] = model_name
                
                # 완료되는 대로 결과 수집
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result_info = future.result()
                        group_results[model_name] = result_info
                        all_results[model_name] = result_info
                        if result_info:
                            time_info = result_info["time_info"]
                            print(f"모델 {model_name}에 대한 self-refinement 완료 (처리 시간: {time_info['processing_time_minutes']:.2f}분)")
                        else:
                            print(f"모델 {model_name}에 대한 self-refinement 실패")
                    except Exception as e:
                        print(f"모델 {model_name}에 대한 self-refinement 실행 오류: {e}")
                        group_results[model_name] = None
                        all_results[model_name] = None
        else:
            # 순차적으로 실행
            for model_name in model_group:
                # 이 모델의 결과 파일 찾기
                results_filename = f"gsm8k_{model_name}.json"
                results_path = os.path.join(RESULTS_DIR, results_filename)
                
                if not os.path.exists(results_path):
                    print(f"모델 {model_name}에 대한 결과 파일을 찾을 수 없음: {results_path}")
                    continue
                    
                try:
                    result_info = run_self_refinement_for_model(
                        model_name=model_name,
                        results_dir=RESULTS_DIR,
                        output_dir=OUTPUT_DIR,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P
                    )
                    group_results[model_name] = result_info
                    all_results[model_name] = result_info
                except Exception as e:
                    print(f"모델 {model_name}에 대한 self-refinement 실행 오류: {e}")
                    group_results[model_name] = None
                    all_results[model_name] = None
                    
        # 각 그룹 완료 후 중간 결과 보고
        print(f"\n그룹 {group_idx+1} self-refinement 완료. 결과 요약:")
        time_summary = {}
        for model_name, result_info in group_results.items():
            if result_info:
                time_info = result_info["time_info"]
                path = result_info["path"]
                time_summary[model_name] = time_info
                print(f" {model_name}:")
                print(f"   - 처리 시간: {time_info['processing_time_seconds']:.2f}초 ({time_info['processing_time_minutes']:.2f}분)")
                print(f"   - 평균 문제당 시간: {time_info['avg_time_per_question']:.2f}초")
                print(f"   - 출력 경로: {path}")
            else:
                print(f" {model_name}: 실패")
    
    # 모든 그룹 완료 후 최종 결과 보고
    print("\n모든 그룹 self-refinement 완료. 결과 요약:")
    
    # 모든 모델 처리 시간 정보를 하나의 JSON 파일로 저장
    all_time_summary = {}
    
    for model_name, result_info in all_results.items():
        if result_info:
            time_info = result_info["time_info"]
            path = result_info["path"]
            all_time_summary[model_name] = time_info
            print(f" {model_name}:")
            print(f"   - 처리 시간: {time_info['processing_time_seconds']:.2f}초 ({time_info['processing_time_minutes']:.2f}분)")
            print(f"   - 평균 문제당 시간: {time_info['avg_time_per_question']:.2f}초")
            print(f"   - 출력 경로: {path}")
        else:
            print(f" {model_name}: 실패")
    
    # 시간 정보 요약 저장
    time_summary_path = os.path.join(OUTPUT_DIR, "refinement_time_summary.json")
    with open(time_summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_time_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"시간 요약 정보는 {time_summary_path}에 저장되었습니다.")