import pandas as pd
import os
import chardet
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')


def detect_file_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding']


def read_medical_csv(file_path):
    """
    读取医疗CSV文件，自动处理编码问题
    """
    encodings_to_try = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig', 'latin1', 'cp1252']

    try:
        detected_encoding = detect_file_encoding(file_path)
        if detected_encoding:
            encodings_to_try.insert(0, detected_encoding)
    except:
        pass

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, Exception):
            continue

    try:
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    except:
        try:
            return pd.read_csv(file_path, encoding='gbk', errors='ignore')
        except Exception as e:
            return None


def clean_medical_data(df, department_name):
    """
    清洗医疗数据
    """
    original_shape = df.shape

    # 删除完全为空的行
    df = df.dropna(how='all')

    # 重命名列
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['title', '标题', '主题']):
            column_mapping[col] = 'title'
        elif any(x in col_lower for x in ['question', '问题', 'ask', '询问']):
            column_mapping[col] = 'question'
        elif any(x in col_lower for x in ['answer', '答案', '回答', 'reply']):
            column_mapping[col] = 'answer'
        elif any(x in col_lower for x in ['department', '科室', 'dept']):
            column_mapping[col] = 'department'

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # 确保有必要的列
    if not any(col in df.columns for col in ['title', 'question', 'answer']):
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'title', df.columns[1]: 'question'})
            if len(df.columns) >= 3:
                df = df.rename(columns={df.columns[2]: 'answer'})

    # 处理缺失值
    if 'question' in df.columns:
        df = df.dropna(subset=['question'])
    if 'answer' in df.columns:
        df = df.dropna(subset=['answer'])

    # 去除空白字符
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    # 过滤过短的回答
    if 'answer' in df.columns:
        df = df[df['answer'].str.len() > 5]

    # 去重
    if 'question' in df.columns:
        df = df.drop_duplicates(subset=['question'], keep='first')

    # 添加科室信息和来源标识
    if 'department' not in df.columns:
        df['department'] = department_name
    df['source'] = f'medical_dialogue_{department_name}'

    cleaned_shape = df.shape
    return df


def load_all_medical_data(data_dir="Data_数据", sample_size=None):
    """
    加载所有医疗数据
    """
    if not os.path.exists(data_dir):
        return None

    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    all_dataframes = []
    total_records = 0

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)

        csv_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(folder_path, file))

        if not csv_files:
            continue

        for csv_file in csv_files:
            df = read_medical_csv(csv_file)

            if df is None or df.empty:
                continue

            df_clean = clean_medical_data(df, folder)

            if df_clean.empty:
                continue

            if sample_size and len(df_clean) > sample_size:
                df_clean = df_clean.sample(sample_size, random_state=42)

            all_dataframes.append(df_clean)
            total_records += len(df_clean)

    if not all_dataframes:
        return None

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df


def create_chunks_for_rag(df, chunk_type="qa_pair"):
    """
    为RAG系统创建chunks
    """
    chunks = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理chunks"):
        if chunk_type == "qa_pair":
            chunk = {
                "id": idx,
                "text": f"问题：{row.get('question', '')}\n答案：{row.get('answer', '')}",
                "metadata": {
                    "department": row.get('department', ''),
                    "source": row.get('source', ''),
                    "question": row.get('question', ''),
                    "answer": row.get('answer', ''),
                    "title": row.get('title', '') if 'title' in row else ''
                }
            }
        elif chunk_type == "question_only":
            chunk = {
                "id": idx,
                "text": row.get('question', ''),
                "content": row.get('answer', ''),
                "metadata": {
                    "department": row.get('department', ''),
                    "source": row.get('source', ''),
                    "full_qa": f"问题：{row.get('question', '')}\n答案：{row.get('answer', '')}",
                    "title": row.get('title', '') if 'title' in row else ''
                }
            }
        else:
            chunk = {
                "id": idx,
                "text": f"{row.get('title', '')} {row.get('question', '')}",
                "content": row.get('answer', ''),
                "metadata": {
                    "department": row.get('department', ''),
                    "source": row.get('source', ''),
                    "full_qa": f"标题：{row.get('title', '')}\n问题：{row.get('question', '')}\n答案：{row.get('answer', '')}",
                }
            }

        chunks.append(chunk)

    return chunks


def save_processed_data(df, output_file="medical_qa_processed.csv"):
    """保存处理后的数据"""
    df.to_csv(output_file, index=False, encoding='utf-8')
    return output_file


if __name__ == "__main__":
    combined_df = load_all_medical_data(data_dir="Data_数据", sample_size=None)

    if combined_df is None:
        print("数据加载失败")
        exit(1)

    save_processed_data(combined_df, "medical_qa_all.csv")

    chunk_type = "question_only"
    chunks = create_chunks_for_rag(combined_df, chunk_type=chunk_type)

    with open("medical_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("数据处理完成")