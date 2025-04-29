import os
import csv
import glob
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def find_majority_better_weights(experiment_dir: str, baseline_weights: List[int] = [1, 1, 1, 1, 1], threshold_percent: float = 75.0):
    """
    대부분의 CSV 파일에서 기준 가중치([1,1,1,1,1])보다 성능이 좋은 가중치 조합을 찾습니다.
    
    Args:
        experiment_dir: 실험 결과가 저장된 디렉토리
        baseline_weights: 비교 기준이 되는 가중치 조합
        threshold_percent: 향상을 보여야 하는 모델의 최소 비율 (%)
    
    Returns:
        대부분의 파일에서 기준보다 나은 가중치 조합 목록
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(experiment_dir, "non_increasing_weight_search_*.csv"))
    
    if not csv_files:
        print(f"경고: {experiment_dir}에서 CSV 파일을 찾을 수 없습니다!")
        return []
    
    print(f"총 {len(csv_files)}개 CSV 파일을 분석합니다.")
    
    # 임계값 계산 (모델 수 기준)
    threshold_count = max(1, int((threshold_percent / 100) * len(csv_files)))
    print(f"기준: 최소 {threshold_count}개/{len(csv_files)}개 모델({threshold_percent:.1f}%)에서 향상된 가중치 조합 찾기")
    
    # 모델별 기준 정확도
    baseline_accuracies = {}
    
    # 모델별 기준보다 나은 가중치 집합
    better_weights_by_model = {}
    
    # 가중치별 개선 모델 수 카운트
    weight_improvement_count = defaultdict(int)
    
    # 모든 가중치 조합 추적
    all_weights = set()
    
    # 각 CSV 파일 처리
    for csv_file in csv_files:
        # 파일명에서 모델명 추출
        model_name = os.path.basename(csv_file).replace("non_increasing_weight_search_", "").replace(".csv", "")
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
        
        # 기준보다 나은 가중치 찾기
        better_weights = {weights for weights, acc in weights_accuracy.items() if acc > baseline_accuracy}
        better_weights_by_model[model_name] = better_weights
        
        # 각 가중치 조합이 이 모델에서 향상되었는지 카운트
        for weights in better_weights:
            weight_improvement_count[weights] += 1
        
        print(f"모델 {model_name}에서 기준보다 나은 가중치: {len(better_weights)}개")
    
    # 대다수 모델에서 향상된 가중치 찾기
    majority_better_weights = [weights for weights, count in weight_improvement_count.items() 
                              if count >= threshold_count]
    
    # 향상도 순으로 정렬하기 위한 정보 수집
    weight_performances = {}
    
    for weights in majority_better_weights:
        performances = {}
        for model_name in better_weights_by_model.keys():
            # 해당 가중치 조합의 정확도 찾기
            csv_file = os.path.join(experiment_dir, f"non_increasing_weight_search_{model_name}.csv")
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    weights_str = row['weights'].strip('[]').split(',')
                    row_weights = tuple(int(w.strip()) for w in weights_str)
                    
                    if row_weights == weights:
                        accuracy = float(row['accuracy'])
                        baseline = baseline_accuracies[model_name]
                        improvement = accuracy - baseline
                        performances[model_name] = (accuracy, improvement)
                        break
        
        # 총합 및 평균 개선도 계산 (모든 모델 대상)
        total_improvement = sum(imp for _, imp in performances.values())
        avg_improvement = total_improvement / len(performances) if performances else 0
        
        # 향상된 모델 수
        improved_count = weight_improvement_count[weights]
        improved_percent = (improved_count / len(better_weights_by_model)) * 100
        
        weight_performances[weights] = {
            'model_performances': performances,
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'improved_count': improved_count,
            'improved_percent': improved_percent
        }
    
    # 평균 개선도 기준으로 정렬
    sorted_weights = sorted(
        weight_performances.items(), 
        key=lambda x: x[1]['avg_improvement'], 
        reverse=True
    )
    
    # 결과 출력
    print(f"\n=== {threshold_percent:.1f}% 이상의 모델에서 기준보다 나은 가중치 조합 ({len(sorted_weights)}개) ===")
    if not sorted_weights:
        print("조건을 만족하는 가중치 조합이 없습니다.")
        return []
    
    print("(평균 개선도 기준 내림차순 정렬)")
    print("=" * 80)
    print(f"{'가중치 조합':<20} {'향상 모델':<15} {'평균 개선도':<15} {'총 개선도':<15} {'모델별 변화'}")
    print("-" * 80)
    
    for weights, perf in sorted_weights:
        weights_str = str(list(weights))
        improved = f"{perf['improved_count']}/{len(better_weights_by_model)} ({perf['improved_percent']:.1f}%)"
        avg_imp = perf['avg_improvement']
        total_imp = perf['total_improvement']
        
        print(f"{weights_str:<20} {improved:<15} {avg_imp:.6f}   {total_imp:.6f}   ", end="")
        
        # 모델별 개선도 출력 (간결하게)
        model_perfs = []
        for model, (acc, imp) in perf['model_performances'].items():
            # 개선 여부에 따라 +/- 기호 추가
            sign = '+' if imp > 0 else ''
            model_perfs.append(f"{model}:{sign}{imp:.6f}")
        
        print(", ".join(model_perfs[:3]) + (", ..." if len(model_perfs) > 3 else ""))
    
    print("=" * 80)
    
    # 결과를 CSV로 저장
    output_csv = os.path.join(experiment_dir, f"majority_{int(threshold_percent)}percent_better_weights.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'avg_improvement', 'total_improvement'] + list(better_weights_by_model.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for weights, perf in sorted_weights:
            row = {
                'weights': str(list(weights)),
                'improved_count': perf['improved_count'],
                'improved_percent': perf['improved_percent'],
                'avg_improvement': perf['avg_improvement'],
                'total_improvement': perf['total_improvement']
            }
            
            # 모델별 개선도 추가
            for model, (acc, imp) in perf['model_performances'].items():
                row[model] = imp
            
            writer.writerow(row)
    
    print(f"\n결과가 {output_csv}에 저장되었습니다.")
    
    # 상세 정보 CSV 저장 (모든 행)
    detailed_csv = os.path.join(experiment_dir, f"majority_{int(threshold_percent)}percent_details.csv")
    with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
        models = list(better_weights_by_model.keys())
        fieldnames = ['weights', 'improved_count', 'improved_percent', 'avg_improvement']
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
    experiment_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/SVAMP/Results/ccqa_t5_result/precision_similar/weight_experiments"
    
    # 기준 가중치 지정 (기본값 [1,1,1,1,1])
    baseline_weights = [1, 1, 1, 1, 1]
    
    # 임계값 설정 (몇 퍼센트의 모델에서 향상을 보여야 하는지)
    threshold_percent = 75.0  # 75%
    
    # 대다수 모델에서 향상된 가중치 찾기
    majority_weights = find_majority_better_weights(experiment_dir, baseline_weights, threshold_percent)
    
    # 추천 가중치 (상위 3개)
    if majority_weights:
        print("\n추천 가중치 조합 (상위 3개):")
        for i, weights in enumerate(majority_weights[:3], 1):
            print(f"{i}. {weights}")
    
    # 임계값을 점진적으로 낮추면서 다시 시도
    if not majority_weights:
        for new_threshold in [70.0, 60.0, 50.0]:
            print(f"\n\n임계값을 {new_threshold}%로 낮추어 다시 시도합니다.")
            majority_weights = find_majority_better_weights(experiment_dir, baseline_weights, new_threshold)
            if majority_weights:
                print(f"\n{new_threshold}% 임계값으로 찾은 추천 가중치 조합 (상위 3개):")
                for i, weights in enumerate(majority_weights[:3], 1):
                    print(f"{i}. {weights}")
                break