import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 데이터 준비
data = [
    {"model": "llama-1b", "CoT": 27.85, "SC": 35.08, "CCQA": 30.99, "T5": 10.07, "extract_place": "first", "extract_way": "regex"},
    {"model": "llama-3b", "CoT": 52.10, "SC": 62.18, "CCQA": 47.42, "T5": 30.81, "extract_place": "first", "extract_way": "regex"},
    {"model": "qwen-0.5b", "CoT": 12.04, "SC": 9.91, "CCQA": 12.04, "T5": 44.22, "extract_place": "first", "extract_way": "regex"},
    {"model": "qwen-1.5b", "CoT": 15.97, "SC": 14.50, "CCQA": 10.04, "T5": 62.32, "extract_place": "first", "extract_way": "regex"},
    {"model": "qwen-3b", "CoT": 49.55, "SC": 51.19, "CCQA": 47.42, "T5": 69.28, "extract_place": "first", "extract_way": "regex"},
    {"model": "llama-1b", "CoT": 1.43, "SC": 5.89, "CCQA": 13.50, "T5": 10.07, "extract_place": "last", "extract_way": "regex"},
    {"model": "llama-3b", "CoT": 8.03, "SC": 17.46, "CCQA": 16.62, "T5": 30.81, "extract_place": "last", "extract_way": "regex"},
    {"model": "qwen-0.5b", "CoT": 0.57, "SC": 0.037, "CCQA": 5.16, "T5": 44.22, "extract_place": "last", "extract_way": "regex"},
    {"model": "qwen-1.5b", "CoT": 9.25, "SC": 12.12, "CCQA": 8.60, "T5": 62.32, "extract_place": "last", "extract_way": "regex"},
    {"model": "qwen-3b", "CoT": 49.55, "SC": 51.19, "CCQA": 41.36, "T5": 69.28, "extract_place": "last", "extract_way": "regex"},
]

# DataFrame으로 변환
df = pd.DataFrame(data)

# 데이터 분할 - 추출 위치별
first_data = df[df['extract_place'] == 'first']
last_data = df[df['extract_place'] == 'last']

# 시각화 함수 정의
def create_method_comparison_plots(include_t5=True):
    # 전체 그림 설정
    plt.figure(figsize=(20, 16))
    plt.suptitle('추출 위치별 메소드 성능 비교', fontsize=22, y=0.98)
    
    # 그리드 설정
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.3)
    
    # 메소드 정의
    methods = ['CoT', 'SC', 'CCQA']
    if include_t5:
        methods.append('T5')
    
    # 색상 정의
    colors = {'CoT': '#8884d8', 'SC': '#82ca9d', 'CCQA': '#ffc658', 'T5': '#ff8042'}
    
    # 1. First 추출 위치 그래프
    ax1 = plt.subplot(gs[0])
    x = np.arange(len(first_data))
    width = 0.2  # 바 너비
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax1.bar(x + offset, first_data[method], width, label=method, color=colors[method])
    
    ax1.set_ylabel('accuracy (%)', fontsize=12)
    ax1.set_title('First extract', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(first_data['model'])
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 값 레이블 추가
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        for j, v in enumerate(first_data[method]):
            ax1.text(j + offset, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Last 추출 위치 그래프
    ax2 = plt.subplot(gs[1])
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax2.bar(x + offset, last_data[method], width, label=method, color=colors[method])
    
    ax2.set_ylabel('accuracy (%)', fontsize=12)
    ax2.set_title('Last extract', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(last_data['model'])
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 값 레이블 추가
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        for j, v in enumerate(last_data[method]):
            ax2.text(j + offset, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. First-Last 차이 히트맵
    ax3 = plt.subplot(gs[2])
    
    # 차이 계산을 위한 데이터 준비
    diff_data = pd.DataFrame(columns=['model'] + methods)
    
    for model in first_data['model'].unique():
        row = {'model': model}
        for method in methods:
            first_val = first_data[first_data['model'] == model][method].values[0]
            last_val = last_data[last_data['model'] == model][method].values[0]
            row[method] = first_val - last_val
        diff_data = pd.concat([diff_data, pd.DataFrame([row])], ignore_index=True)
    
    # 피벗 테이블 생성
    diff_pivot = diff_data.set_index('model')
    
    # 히트맵 그리기
    sns.heatmap(diff_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('first-last diffenence', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # 파일로 저장
    save_filename = 'method_comparison_with_t5.png' if include_t5 else 'method_comparison_without_t5.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    return plt

# 추가적인 분석을 위한 함수
def create_best_method_analysis():
    plt.figure(figsize=(18, 10))
    
    # 1. First 위치에서 각 모델별 최고 성능 메소드
    plt.subplot(2, 2, 1)
    
    first_best_method = pd.DataFrame(columns=['model', 'best_method', 'accuracy'])
    for model in first_data['model'].unique():
        model_data = first_data[first_data['model'] == model]
        methods = ['CoT', 'SC', 'CCQA', 'T5']
        best_method = max(methods, key=lambda m: model_data[m].values[0])
        best_accuracy = model_data[best_method].values[0]
        first_best_method = pd.concat([first_best_method, 
                                     pd.DataFrame([{'model': model, 'best_method': best_method, 'accuracy': best_accuracy}])], 
                                     ignore_index=True)
    
    # 바 차트로 표시
    bars = plt.bar(first_best_method['model'], first_best_method['accuracy'], color='skyblue')
    
    # 메소드 이름 표시
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 5, 
                 first_best_method['best_method'].iloc[i], 
                 ha='center', va='bottom', color='black', fontweight='bold')
    
    plt.title('first extract best method')
    plt.ylabel('정확도 (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Last 위치에서 각 모델별 최고 성능 메소드
    plt.subplot(2, 2, 2)
    
    last_best_method = pd.DataFrame(columns=['model', 'best_method', 'accuracy'])
    for model in last_data['model'].unique():
        model_data = last_data[last_data['model'] == model]
        methods = ['CoT', 'SC', 'CCQA', 'T5']
        best_method = max(methods, key=lambda m: model_data[m].values[0])
        best_accuracy = model_data[best_method].values[0]
        last_best_method = pd.concat([last_best_method, 
                                    pd.DataFrame([{'model': model, 'best_method': best_method, 'accuracy': best_accuracy}])], 
                                    ignore_index=True)
    
    # 바 차트로 표시
    bars = plt.bar(last_best_method['model'], last_best_method['accuracy'], color='lightgreen')
    
    # 메소드 이름 표시
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 5, 
                 last_best_method['best_method'].iloc[i], 
                 ha='center', va='bottom', color='black', fontweight='bold')
    
    plt.title('Last extract best model')
    plt.ylabel('정확도 (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. 각 메소드별 First-Last 평균 차이
    plt.subplot(2, 1, 2)
    
    methods = ['CoT', 'SC', 'CCQA', 'T5']
    avg_diffs = []
    
    for method in methods:
        first_avg = first_data[method].mean()
        last_avg = last_data[method].mean()
        diff = first_avg - last_avg
        avg_diffs.append({'method': method, 'difference': diff, 'first_avg': first_avg, 'last_avg': last_avg})
    
    avg_diffs_df = pd.DataFrame(avg_diffs)
    
    # 그룹화된 바 차트로 표시
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, avg_diffs_df['first_avg'], width, label='First 평균', color='royalblue')
    plt.bar(x + width/2, avg_diffs_df['last_avg'], width, label='Last 평균', color='lightcoral')
    
    # 차이 표시
    for i, row in avg_diffs_df.iterrows():
        plt.text(i, max(row['first_avg'], row['last_avg']) + 3, 
                 f'차이: {row["difference"]:.1f}%p', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.title('각 메소드별 First-Last 평균 성능 차이')
    plt.ylabel('평균 정확도 (%)')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('method_best_analysis.png', dpi=300, bbox_inches='tight')
    
    return plt

# 실행
plot1 = create_method_comparison_plots(include_t5=True)
plot2 = create_method_comparison_plots(include_t5=False)
plot3 = create_best_method_analysis()

plot1.show()
plot2.show()
plot3.show()