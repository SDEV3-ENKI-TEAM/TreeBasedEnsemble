import pandas as pd

# CSV 파일 경로
input_path = "labels.csv"
output_path = "eval_labels_sorted.csv"

# CSV 읽기
df = pd.read_csv(input_path)

# traceID 기준 오름차순 정렬
df_sorted = df.sort_values(by="traceID", ascending=True)

# 정렬된 결과 저장
df_sorted.to_csv(output_path, index=False)

print(f"✅ traceID 기준으로 정렬된 파일이 '{output_path}'로 저장되었습니다.")
print(f"총 {len(df_sorted)}개의 행이 정렬되었습니다.")
