# MentalHealthRAG

## LLM 與 Embedding 設定

MentalHealthRAG 使用 Google Generative AI 提供的 LLM 來進行文本嵌入與分析：

- **Embedding 模型**：`text-embedding-004`
- **生成式模型**：`gemini-1.5-flash-latest`
- **API 提供者**：Google Generative AI

# CSV 欄位需求

每一行資料將包含以下欄位：

- **標題**：文章標題
- **內文**：文章內容
- **日期**：文章日期
- **作者**：文章作者
- **label**：該文章的分類（0,1）


## 如何使用 MentalHealthRAG

1. **建立向量資料庫**：
    - 首先，將所有訓練資料進行處理，並使用 LLM（大語言模型）生成相應的嵌入向量（embedding）。這些嵌入向量將被存入向量資料庫中，這樣可以進行後續的檢索工作。

2. **準備 Prompt 文件**：
    - 撰寫並準備好對應的 `prompt.txt` 文件，這個文件將用於引導模型進行特定的分析。文件中應包含如何理解和處理待分析文章的提示與指引。
    
    - 以下為範例
        ```yaml
        請根據以下專家的標準，判斷給定文章的分數。

        專家定義標準：

        ## example

        1：尋求協助。
        0：不尋求協助。

        以上為專家定義標準，請依照以下步驟進行評估：

        輸入測試文章：

        {article}

        輸出格式：

        請給出數值。

        請僅輸出數值，無需額外解釋。
        ```

3. **預測文章的情感分析**：
    - 將待分析的文章輸入系統。系統會根據內部的檢索機制從向量資料庫中檢索出與文章最相關的內容，並將這些相關資訊作為上下文提供給 LLM 進行分析。

4. **生成結果**：
    - LLM 會根據檢索到的相關資料和 `prompt.txt` 文件的指引，對文章進行分析，最終給出對應的結果，通常以分數的形式呈現。


## 快速開始

- 若要插入資料到向量資料庫，MODE = write
- 反之，MODE = read

1. 修改 .env
    ```yaml
    DATASET="SS22" # your dataset name
    MODE='read' # write / read  to qdrant

    GEMINI_API_KEY = ["your-api-key"]
    QDRANT_HOST="your-qdant-host-ip" # e.g. 22.22.11.22

    FOLDER_PATH="your folder path" # project folder
    DATA_PATH="data/${DATASET}" # data folder in project folder


    # 設定qdrant重試參數
    MAX_RETRIES=3  # 最多重試次數
    TIMEOUT=30  # 設定超時時間（秒）
    DELAY=5  # 每次重試間隔（秒）

    # 若無將資料集做kfold，預設為1
    FOLDS=1

    # 若有多個emini api key，每五個推論要求，就會換下一個gemini api key，防止key被封
    GEMINI_REQUEST_COUNT=5

    MAPPING='{0: 1, 1: 2}' # 將分數與label配對
    ```

2. python main

    ```bash
    python main 
    ```