import pandas as pd
# from langchain.vectorstores import Qdrant
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import dotenv_values
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import time
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 確保 Collection 存在
def ensure_collection(create=False):
    if collection_name not in [c.name for c in client.get_collections().collections] and create:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE), 
            )
            print(f"建立 collection : {collection_name}")
            return True
    else:
        print(f"collection 已存在")
        return False

def load_llm(gemini_api_key):
    # 載入 embedding 和生成器，使用當前的 API 金鑰
    embedding_llm = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=gemini_api_key
    )

    generator_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", google_api_key=gemini_api_key
    )
    print(f"Using API key: {current_key}")

    return embedding_llm, generator_llm

# 讀取並處理 CSV 檔案
def import_csv_to_qdrant(csv_path):
    
    embedding_llm,generator_llm = load_llm(current_key)

    df = pd.read_csv(csv_path, encoding='utf-8')
    documents = [
        Document(
            page_content=f"{row['標題']} {row['text']}",
            metadata={
                "標題": row['標題'],
                "內文": row['text'],
                "日期": row['日期'],
                "作者": row['作者'],
            },
            payload={
                "label": row['label']  # label 存放於 payload
            }
        )
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV")
    ]
    
    # 建立 Qdrant 向量庫，插入資料並顯示進度條
    with tqdm(total=len(documents), desc="Inserting to Qdrant") as pbar:
        qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding_llm,
            url=f"http://{qdrant_host}:{port}",
            collection_name=collection_name,
        )
        pbar.update(len(documents))

    # qdrant_vectorstore = Qdrant.from_documents(
    #     documents, embedding_llm, location=qdrant_host, collection_name=collection_name
    # )
    print(f"成功匯入 {len(documents)} 筆資料！")
    return qdrant

def create_QA_chain(embedding_llm,generator_llm,prompt):
    
    qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_llm
        )

    # 設置檢索器
    retriever = qdrant.as_retriever(search_kwargs={"k": 5}) # 檢索前5個最相似的文檔

    with open(prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    q_template = ChatPromptTemplate.from_template(prompt_template)

    # 建立 QA Chain
    qa_chain = (
        {
            "context": retriever ,
            "article": RunnablePassthrough(),
            # "dataset": lambda x: dataset 
        }
        | q_template
        | generator_llm
        | StrOutputParser()
    )
    return qa_chain

def invoke_with_retry(qa_chain, text, retries=3, delay=5):
    """封裝重試邏輯，處理請求超時或連線錯誤"""
    for attempt in range(retries):
        try:
            # 嘗試執行請求
            result = qa_chain.invoke(text)
            return result  # 如果成功則返回結果
        except Exception as e:
            # 輸出錯誤訊息並嘗試重試
            print(f"Error occurred: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay)  # 等待後再重試
    raise Exception(f"Max retries reached. Failed to process the request after {retries} attempts.")

def eval_model(csv_path,output_path,prompt_path,num_samples):
    prompt = os.path.join(prompt_path,f'{dataset}_prompt.txt')
    global request_count,current_key
    embedding_llm,generator_llm = load_llm(current_key)
    qa_chain = create_QA_chain(embedding_llm,generator_llm,prompt)

    # 讀取 CSV
    if num_samples > 0:
        df = pd.read_csv(csv_path, encoding='utf-8').sample(n=num_samples, random_state=42)
    else:
        df = pd.read_csv(csv_path, encoding='utf-8')

    predictions = []
    labels = df["label"].tolist()  # 取得標籤值

    for text in tqdm(df["text"], desc="Evaluating QA Model"):

        # 每N個請求換一次金鑰
        if request_count >= int(gemini_request_count):
            current_key = gemini_api_keys[(gemini_api_keys.index(current_key) + 1) % len(gemini_api_keys)]
            request_count = 0  # 重設計數器
            embedding_llm,generator_llm = load_llm(current_key)
            qa_chain = create_QA_chain(embedding_llm,generator_llm,prompt)

        # 使用 Agent 處理 text
        # result = qa_chain.invoke(text)
        result = invoke_with_retry(qa_chain,text)

        # 轉換答案格式為數字
        try:
            prediction = int(result)
            if prediction not in mapping_dict:  # 確保符合預期類別
                prediction = 0
        except ValueError:
            prediction = 0  # 預設錯誤答案為 0
        
        prediction = mapping_dict.get(prediction,4)
        predictions.append(prediction)

        # 計數器增長
        request_count += 1
    
    # 儲存結果到 CSV
    result_df = pd.DataFrame({
        "article_id": df['id'],
        "label": labels,
        "prediction": predictions
    })
    result_df_path = os.path.join(output_path,f'{dataset}_fold_{fold}_result.csv')
    
    result_df.to_csv(result_df_path, index=False, encoding='utf-8')
    print(f"結果已儲存至 {result_df_path}")

    # 計算評估指標
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    print(f"模型評估完成！準確度: {accuracy:.2%}, 精確率: {precision:.2%}, 召回率: {recall:.2%}, F1 分數: {f1:.2%}")
    
    # 儲存評估指標到 JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df_path = os.path.join(output_path,f'{dataset}_fold_{fold}_metrics.csv')
    metrics_df.to_csv(metrics_df_path, index=False, encoding='utf-8')
    print(f"評估指標已儲存至 {output_path}")

    return predictions, labels, accuracy, precision, recall, f1

def create_cm(output_path,fold): # 建立混淆矩陣
    
    result_path = os.path.join(output_path,f'{dataset}_fold_{fold}_result.csv')
    # 將數據轉為 DataFrame
    df = pd.read_csv(result_path, encoding='utf-8')

    # 提取 label 和 prediction 欄位
    y_true = df['label']
    y_pred = df['prediction']
    
    # 計算混淆矩陣
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(mapping_dict))))

    # 創建混淆矩陣 DataFrame
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=list(range(len(mapping_dict))), index=list(range(len(mapping_dict))))

    # 將混淆矩陣寫入 CSV
    conf_matrix_df.to_csv(os.path.join(output_path,f'{dataset}_cm_{fold}.csv'))

    # 顯示結果
    print(conf_matrix_df)

def calculate_metrics(conf_matrix): # 以混淆矩陣計算評估分數
    num_classes = conf_matrix.shape[0]
    metrics = {}
    
    total_correct = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()
    overall_accuracy = total_correct / total_samples
    
    for i in range(num_classes):
        TP = conf_matrix[i, i]  # True Positives
        FP = conf_matrix[:, i].sum() - TP  # False Positives
        FN = conf_matrix[i, :].sum() - TP  # False Negatives
        TN = total_samples - (TP + FP + FN)  # True Negatives
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    return overall_accuracy, metrics

def combine_cm(output_path): # 合併kfold的所有混淆矩陣
    file_path = os.path.join(output_path, f'{dataset}_cm_0.csv')# 讀取第一個 fold 的混淆矩陣來初始化 combined_conf_matrix
    combined_conf_matrix = pd.read_csv(file_path, index_col=0).fillna(0)
    
    # 從 fold 1 開始，逐步累加後續的混淆矩陣
    for fold in range(1,folds):
        # 讀取每個 fold 對應的混淆矩陣
        file_path = os.path.join(output_path, f'{dataset}_cm_{fold}.csv')
        conf_matrix = pd.read_csv(file_path, index_col=0).fillna(0)
        
        # 累加混淆矩陣
        combined_conf_matrix += conf_matrix
        # print(f"After adding Fold {fold}:\n{combined_conf_matrix}")
    
    # 將加總後的混淆矩陣寫入新的CSV檔案
    combined_conf_matrix.to_csv(os.path.join(output_path,'combined_confusion_matrix.csv'))

    # 讀取 CSV 文件
    df = pd.read_csv(os.path.join(output_path,'combined_confusion_matrix.csv'), index_col=0).fillna(0)
    conf_matrix = df.values

    # 計算整體準確率與各類別指標
    overall_accuracy, metrics = calculate_metrics(conf_matrix)

    # 過濾掉與文章無關的類別，計算其餘類別的平均
    filtered_metrics = {k: v for k, v in metrics.items() if k != 4}
    num_filtered = len(filtered_metrics)

    avg_metrics = {
        'precision': sum(v['precision'] for v in filtered_metrics.values()) / num_filtered,
        'recall': sum(v['recall'] for v in filtered_metrics.values()) / num_filtered,
        'f1_score': sum(v['f1_score'] for v in filtered_metrics.values()) / num_filtered
    }

    metrics['avg'] = avg_metrics
    metrics['overall_accuracy'] = {'precision': overall_accuracy, 'recall': '', 'f1_score': ''}

    # 輸出結果
    output_df = pd.DataFrame.from_dict(metrics, orient='index')

    # 輸出結果
    output_df = pd.DataFrame.from_dict(metrics, orient='index')
    output_df.to_csv(os.path.join(output_path, 'metrics_output.csv'))

    # 自訂標籤對應
    custom_labels = list(mapping_dict.keys())

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=custom_labels, yticklabels=custom_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_path, 'confusion_matrix.jpg'))
    plt.close()

if __name__ == '__main__': 

    config = dotenv_values(".env")

    gemini_api_keys_str = config.get("GEMINI_API_KEYS", "[]")
    qdrant_host = config.get("QDRANT_HOST")

    dataset = config.get("DATASET")
    folder_path = config.get("FOLDER_PATH")
    data_path = config.get("DATA_PATH")

    prompt_path = os.path.join(folder_path,f"prompts")
    output_path = os.path.join(folder_path,f"out/{dataset}")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(prompt_path, exist_ok=True)
    
    port = 6333
    
    mode = config.get("MODE") # write / read
    gemini_request_count = config.get("GEMINI_REQUEST_COUNT")
    gemini_api_keys = ast.literal_eval(gemini_api_keys_str)# 將字串解析為列表

    mapping_str = config.get("MAPPING")
    mapping_dict = ast.literal_eval(mapping_str)

    # 初始化 Qdrant 客戶端
    client = QdrantClient(qdrant_host, port=port)

    # 設定計數器來追蹤推論次數
    request_count = 0

    # 用來選擇當前使用的 API 金鑰
    current_key = gemini_api_keys[0]

    folds = int(config.get("FOLDS"))
    for fold in range(folds):
        collection_name = f"{dataset}_fold_{fold}"
        train_file_path = os.path.join(folder_path,data_path,f'train_fold_{fold}.csv')
        test_file_path=os.path.join(folder_path,data_path,f"val_fold_{fold}.csv")

        if mode=='read':
            ensure_collection(create=False)
            eval_model(test_file_path,output_path,prompt_path,num_samples=2)
            create_cm(output_path,fold)
        else:
            status = ensure_collection(create=True)
            if status:
                import_csv_to_qdrant(train_file_path)
    
    if mode == 'read':combine_cm(output_path)

