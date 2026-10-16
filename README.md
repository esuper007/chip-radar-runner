# chip-radar 全球籌碼雷達

每天自動抓取台、美、日、港、陸股的公開籌碼資料，計算籌碼分數，並預測 1 季、半年、1 年的上漲機率與漲跌幅區間。結果顯示在 claude.ai 的「籌碼雷達」儀表板，也可以在 Claude 對話中用 `global-chip-analysis` skill 查詢。

```
交易所／主管機關公開資料
        │  GitHub Actions（每個交易日收盤後）
        ▼
Release store-<市場>   歷史資料（每年一個 parquet，不佔 git 歷史）
        │  pipeline.run build
        ▼
分支 dist             當日結果：summary / screen / series（每次覆蓋）
        │  排程工作「籌碼雷達：同步儀表板」（台北 09:20、23:20）
        ▼
claude.ai 儀表板「籌碼雷達」  ＋  Claude skill「global-chip-analysis」
```

## 資料來源

| 市場 | 籌碼資料 |
|---|---|
| 台股 | 證交所、櫃買中心（三大法人、融資融券、借券、外資持股、當沖）、集保（股權分散）、公開資訊觀測站（董監持股與質押） |
| 美股 | FINRA（每日放空量、空單餘額）、SEC（Form 4 內部人、13F、13D/13G、交割失敗） |
| 日股 | JPX（銘柄別信用取引残高、空売り残高報告、投資部門別） |
| 港股 | 港交所（港股通持股、沽空成交）、證監會（淡倉） |
| 陸股 | 上交所、深交所（融資融券）、港交所（陸股通持股）、東方財富／同花順（主力資金、龍虎榜、股東戶數、增減持） |
| 股價 | Yahoo Finance（含息還原） |

## Repo 與權限

- 本 repo（`chip-radar-runner`，公開）只放程式與工作流程；公開 repo 的 GitHub Actions 不計分鐘數。
- 資料放在私有 repo `esuper007/chip-radar`：Release `store-<市場>`（歷史資料與模型）和 `dist` 分支（每日結果）。交易所資料不會公開再散布。
- 需要兩個 Actions Secret：
  - `DATA_TOKEN`：fine-grained 權杖，只能存取 `esuper007/chip-radar`，權限 Contents: Read and write。
  - `SEC_CONTACT`：SEC 要求的聯絡 email；未設定時跳過 SEC 資料。
- 也可以用 Actions variable `DATA_REPO` 改指向其他資料 repo。
- 執行紀錄是公開的，程式只記錄欄位名稱與筆數，不記錄資料數值。

## 工作流程（GitHub → Actions）

| 工作流程 | 何時執行 | 用途 |
|---|---|---|
| market-daily | 每個交易日自動（日 18:30、陸 18:30、港 19:30、台 22:40、美 19:30 當地時間）；也可手動 | 抓當日資料、更新股價、重算分數與預測、發佈到 `dist` |
| backfill | 手動 | 回補歷史資料，超過 5.5 小時會自動接力 |
| weekly-model | 每週六 10:00（台北） | 逐步往前回測，更新儀表板上的模型成績 |
| all-markets | 手動（儀表板「全部重抓」按鈕） | 依序更新五個市場 |

手動執行：GitHub repo →「Actions」→ 選工作流程 →「Run workflow」。
`market-daily` 的 `force` 欄位可填資料集名稱（例如 `inst,daytrade`），強制重新下載最近 7 天。

## 程式結構

| 位置 | 內容 |
|---|---|
| `pipeline/markets/<市場>.py` | 各市場資料抓取與籌碼因子、評分面向與權重 |
| `pipeline/factors.py` | 評分引擎（子分數與綜合分數 0–100）、籌碼訊號 |
| `pipeline/predict.py` | 歷史相似情境（k-NN）與 LightGBM，含逐步往前回測 |
| `pipeline/output.py` | 輸出儀表板用的 JSON |
| `pipeline/store.py` | Release 歷史資料存取（多個工作同時寫入時會安全合併） |
| `dashboard/index.html` | 儀表板頁面原始碼 |
| `skill/global-chip-analysis/` | Claude skill 與指標說明 |
| `tests/synthetic_tw.py` | 產生假資料做離線測試 |

## 已知限制

- 集保股權分散表只公開最新一週，大戶／散戶因子從開始執行那週起累積。
- 日股信用殘高、空売り残高報告的網站只保留近期檔案，歷史從開始執行起累積。
- 陸股主力資金只有當日快照，歷史從開始執行起累積；東方財富有時拒絕海外連線，會改用同花順。
- 台股券商分點（真正的「主力」進出）不在免費公開資料內，目前用「法人＋大戶同向」近似；可另接 FinMind 贊助方案。
- 預測是統計推估，請搭配儀表板上的回測成績判斷可信度。本專案僅供研究，不構成投資建議。
