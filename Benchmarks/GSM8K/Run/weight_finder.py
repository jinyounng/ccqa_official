import os
import csv
import glob
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def find_best_weights_by_total_improvement(experiment_dir: str, baseline_weights: List[int] = [1, 1, 1, 1, 1]):
    """
    모든 모델의 성능 향상 퍼센트를 합산하여 가장 높은 가중치 조합을 찾습니다.
    
    Args:
        experiment_dir: 실험 결과가 저장된 디렉토리
        baseline_weights: 비교 기준이 되는 가중치 조합
    
    Returns:
        총 성능 향상 기준으로 정렬된 가중치 조합 목록
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(experiment_dir, "descending_weight_search_*.csv"))
    
    if not csv_files:
        print(f"경고: {experiment_dir}에서 CSV 파일을 찾을 수 없습니다!")
        return []
    
    print(f"총 {len(csv_files)}개 CSV 파일을 분석합니다.")
    
    # 모델별 기준 정확도
    baseline_accuracies = {}
    
    # 모든 가중치 조합 추적
    all_weights = set()
    
    # 모델별 가중치 정확도 기록
    model_weight_accuracies = {}
    
    # 각 CSV 파일 처리
    for csv_file in csv_files:
        # 파일명에서 모델명 추출
        model_name = os.path.basename(csv_file).replace("descending_weight_search_", "").replace(".csv", "")
        print(f"모델 {model_name} 분석 중...")
        
        # 가중치별 정확도 및 기준 가중치 정확도 찾기
        weights_accuracy = {}
        baseline_accuracy = None
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 가중치 문자열을 리스트로 변환
                weights_str = row['weights'].strip('[]').split(',')
                weights = [int(w.strip()) for w in weights_str]
                
                # 정확도 값 가져오기
                accuracy = float(row['accuracy'])
                
                # 가중치를 튜플로 변환하여 딕셔너리 키로 사용
                weights_tuple = tuple(weights)
                weights_accuracy[weights_tuple] = accuracy
                
                # 모든 가중치 조합 추가
                all_weights.add(weights_tuple)
                
                # 기준 가중치 정확도 기록
                if weights == baseline_weights:
                    baseline_accuracy = accuracy
        
        # 기준 가중치가 없는 경우 처리
        if baseline_accuracy is None:
            print(f"경고: 모델 {model_name}에서 기준 가중치 {baseline_weights}를 찾을 수 없습니다!")
            continue
        
        baseline_accuracies[model_name] = baseline_accuracy
        print(f"모델 {model_name}의 기준 가중치 정확도: {baseline_accuracy:.4f}")
        
        # 모델의 가중치별 정확도 저장
        model_weight_accuracies[model_name] = weights_accuracy
    
    # 모든 가중치 조합에 대한 총 성능 향상 계산
    weight_total_improvements = {}
    weight_performances = {}
    
    for weights in all_weights:
        total_improvement = 0
        performances = {}
        improved_count = 0
        
        for model_name, weight_accuracies in model_weight_accuracies.items():
            if weights in weight_accuracies:
                accuracy = weight_accuracies[weights]
                baseline = baseline_accuracies[model_name]
                improvement = accuracy - baseline
                
                # 성능 향상이 있는 경우만 카운트
                if improvement > 0:
                    improved_count += 1
                
                # 모든 모델의 향상도 합산
                total_improvement += improvement
                performances[model_name] = (accuracy, improvement)
        
        # 향상된 모델 수 비율 계산
        improved_percent = (improved_count / len(model_weight_accuracies)) * 100
        avg_improvement = total_improvement / len(model_weight_accuracies)
        
        weight_total_improvements[weights] = total_improvement
        weight_performances[weights] = {
            'model_performances': performances,
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'improved_count': improved_count,
            'improved_percent': improved_percent
        }
    
    # 총 성능 향상 기준으로 정렬
    sorted_weights = sorted(
        weight_performances.items(), 
        key=lambda x: x[1]['total_improvement'], 
        reverse=True
    )
    
    # 결과 출력
    print(f"\n=== 총 성능 향상 기준 가중치 조합 ({len(sorted_weights)}개) ===")
    if not sorted_weights:
        print("분석할 가중치 조합이 없습니다.")
        return []
    
    print("(총 성능 향상 기준 내림차순 정렬)")
    print("=" * 80)
    print(f"{'가중치 조합':<20} {'향상 모델':<15} {'총 향상도':<15} {'평균 향상도':<15} {'모델별 변화'}")
    print("-" * 80)
    
    for weights, perf in sorted_weights[:20]:  # 상위 20개만 출력
        weights_str = str(list(weights))
        improved = f"{perf['improved_count']}/{len(model_weight_accuracies)} ({perf['improved_percent']:.1f}%)"
        total_imp = perf['total_improvement']
        avg_imp = perf['avg_improvement']
        
        print(f"{weights_str:<20} {improved:<15} {total_imp:.6f}   {avg_imp:.6f}   ", end="")
        
        # 모델별 개선도 출력 (간결하게)
        model_perfs = []
        for model, (acc, imp) in perf['model_performances'].items():
            # 개선 여부에 따라 +/- 기호 추가
            sign = '+' if imp > 0 else ''
            model_perfs.append(f"{model}:{sign}{imp:.6f}")
        
        print(", ".join(model_perfs[:3]) + (", ..." if len(model_perfs) > 3 else ""))
    
    print("=" * 80)
    
    # 결과를 CSV로 저장
    output_csv = os.path.join(experiment_dir, "total_improvement_ranked_weights.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'total_improvement', 'avg_improvement'] + list(model_weight_accuracies.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for weights, perf in sorted_weights:
            row = {
                'weights': str(list(weights)),
                'improved_count': perf['improved_count'],
                'improved_percent': perf['improved_percent'],
                'total_improvement': perf['total_improvement'],
                'avg_improvement': perf['avg_improvement']
            }
            
            # 모델별 개선도 추가
            for model, (acc, imp) in perf['model_performances'].items():
                row[model] = imp
            
            writer.writerow(row)
    
    print(f"\n결과가 {output_csv}에 저장되었습니다.")
    
    # 상세 정보 CSV 저장 (모든 행)
    detailed_csv = os.path.join(experiment_dir, "total_improvement_details.csv")
    with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
        models = list(model_weight_accuracies.keys())
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'total_improvement', 'avg_improvement']
        for model in models:
            fieldnames.append(f"{model}_accuracy")
            fieldnames.append(f"{model}_improvement")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for weights, perf in sorted_weights:
            row = {
                'weights': str(list(weights)),
                'improved_count': perf['improved_count'],
                'improved_percent': perf['improved_percent'],
                'total_improvement': perf['total_improvement'],
                'avg_improvement': perf['avg_improvement']
            }
            
            # 모델별 정확도와 개선도 추가
            for model, (acc, imp) in perf['model_performances'].items():
                row[f"{model}_accuracy"] = acc
                row[f"{model}_improvement"] = imp
            
            writer.writerow(row)
    
    print(f"상세 정보가 {detailed_csv}에 저장되었습니다.")
    
    return [list(weights) for weights, _ in sorted_weights]

if __name__ == "__main__":
    # 실험 결과 디렉토리 지정
    experiment_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_0_result/roberta_precision_similar/descending_weight_experiments"
    
    # 기준 가중치 지정 (기본값 [1,1,1,1,1])
    baseline_weights = [1, 1, 1, 1, 1]
    
    # 총 성능 향상 기준으로 가장 좋은 가중치 찾기
    best_weights = find_best_weights_by_total_improvement(experiment_dir, baseline_weights)
    
    # 추천 가중치 (상위 5개)
    if best_weights:
        print("\n추천 가중치 조합 (상위 5개):")
        for i, weights in enumerate(best_weights[:5], 1):
            print(f"{i}. {weights}")