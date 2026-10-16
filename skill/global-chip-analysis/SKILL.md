---
name: global-chip-analysis
description: 全球股票籌碼分析（台、美、日、港、陸股）：法人、主力、大戶散戶、董監內部人、融資融券、借券放空等籌碼面解讀，並給出 1 季、半年、1 年的漲跌機率與區間。當使用者問某檔股票或某個市場的籌碼、法人動向、大戶持股、融資融券、空單、籌碼排行，或要更新籌碼雷達儀表板時使用。
---

# 全球籌碼分析

程式與資料分放兩個 GitHub repo：
- `esuper007/chip-radar-runner`（公開）：程式（`pipeline/`）與 GitHub Actions 工作流程。公開 repo 的 Actions 不計分鐘數。這裡不放任何資料、email 或權杖。
- `esuper007/chip-radar`（私有）：資料倉庫。Release `store-<市場>` 是歷史資料與模型，`dist` 分支是每天算好的結果，每個市場一個資料夾 `<tw|us|jp|hk|cn>/`。
- 工作流程用 runner repo 的 Secret `DATA_TOKEN`（只能讀寫 `chip-radar` 內容的權杖）存取私有 repo；`SEC_CONTACT` 是 SEC 要求的聯絡 email。權杖失效時，所有工作會在第一步「Check secrets」或存取資料時失敗，要請使用者重新產生並更新 Secret。
- 儀表板 artifact「籌碼雷達」（https://claude.ai/artifact/TR1wdSdupeQtdjmJ9PsaoF）顯示的就是 `dist` 的內容。

## 1. 取得最新結果

```bash
git clone --depth 1 --branch dist https://github.com/esuper007/chip-radar /tmp/chip-dist
```
若 clone 失敗（403/404），先用 `add_repo`（owner `esuper007`、repo `chip-radar`）把 repo 加進這個工作階段再 clone。

每個市場的檔案（副檔名是 .js，內容是純 JSON）：
| 檔案 | 內容 |
|---|---|
| `summary.js` | 資料日 `asof`、各資料集最新日期 `freshness`、資料品質報告 `quality`（`checks` 各項檢查是否通過、`sources` 每個資料集的來源與是否官方）、前/後 30 名、市場層級籌碼序列、模型回測成績 `metrics`、因子中文名稱 `labels` |
| `screen.js` | `asof` 與 `rows`：每檔一列。`c` 代號、`n` 名稱、`i` 產業、`x` 上市別、`liq` 是否流動性足夠、`px` 收盤、`sc` 綜合分數、`sub` 子分數、`f` 關鍵因子（名稱見 `summary.labels`）、`sig` 訊號、`fc` 預測 |
| `series_<k>.js` | 流動性足夠的股票：近 130 個交易日的還原股價與分數（`dates`），籌碼序列為每週一點（`wdates`）；`k = sum(ord(ch) for ch in 代號) % 8` |
| `metrics.json` | 每週六的逐步往前回測結果（與 `summary.metrics` 相同） |
| `_logs/<市場>.log` | 最近一次執行的紀錄，排查資料問題時先看這裡 |

`fc` 的結構：`fc.{an|ml|is}.{1q|6m|1y}.{p_up,q10,q50,q90}`。`an` 是歷史相似情境（k-NN），`ml` 是 LightGBM，`is` 是樣本內外模型（前 70% 期間建模、後段樣本外驗證，用驗證過的同一個模型帶入目前籌碼預測，區間取自樣本外誤差；成績與係數在 `summary.metrics.<期間>.is`）；1q≈63、6m≈126、1y≈252 個交易日；報酬是含息還原的百分比。

`summary.metrics.<1q|6m|1y>` 的結構（每個成績都有 `n` 樣本數、`base_rate_up` 同期實際上漲比例、`hit_rate` 命中率、`auc`、`rank_ic` 排名 IC、`top_minus_bottom_pct` 前後五分位報酬差、`coverage_80` 80% 區間實際覆蓋率）：
- `ml`、`analog`：每年一折的逐步往前回測（每週六的 weekly-model 產生；尚未跑過時不存在）。
- `score_only`：只用綜合分數排序的 IC 與報酬差。
- `is`：`in_sample`、`out_of_sample` 兩組成績、`split` 兩段的起訖日、`coef` 係數（因子從全市場最低排到最高時預期報酬增減的百分點，名稱用 `summary.labels`）。
- `summary.metrics.group_ic`：各子分數對 1 季報酬的排名 IC（同樣由每週回測產生）。

用 Python 讀取，不要把整個檔案印到對話裡：
```python
import json; s = json.load(open('/tmp/chip-dist/tw/screen.js'))
row = next(r for r in s['rows'] if r['c'] == '2330')
```

## 2. 回答個股問題的結構

1. **一句話結論**：籌碼面偏多、偏空或中性，附綜合分數與子分數（0–100，是全市場百分位，50 為中性）。
2. **證據**：挑 3–5 個最關鍵的因子，用 `summary.labels` 的中文名稱，引用數字與方向。台股依序看法人（外資、投信、自營商）、大戶與散戶（集保分級）、信用（融資融券、當沖）、借券放空、董監（持股、質押）。各市場指標的完整說明在 repo 的 `skill/global-chip-analysis/references/indicators.md`（需要時 `git clone --depth 1 https://github.com/esuper007/chip-radar-runner /tmp/chip-main` 後讀取）。
3. **預測**：分列 1 季、半年、1 年，三種方法的上漲機率、中位數與 10–90% 區間。**一定要附上可信度**：引用 `summary.metrics` 中該期間的命中率與「基準」（`base_rate_up`）、排名 IC；`ml`/`analog` 回測還沒跑過時，改用 `is.out_of_sample` 的成績。命中率沒有明顯高於基準時要直接說明。
4. **接下來觀察什麼**：2 個具體的籌碼變化。
5. 結尾一句：這是根據公開資料的統計推估，不構成投資建議。

**先看資料品質**：回答前檢查 `summary.quality`（`n_fail` 未通過項數、`checks` 每項的 `name`、`ok`、`detail`）；有未通過的項目且與該股票相關（例如融資合計不符、收盤價差異、缺漏日期）時，要在回答開頭說明，必要時降低結論的把握度。引用第三方來源（`sources` 中 `official: false`，例如陸股主力資金）時要註明。

缺資料時照實說明，不要用記憶中的數字補。例如：集保分級從開始執行那週起才累積；預測欄位為空代表已知結果的歷史樣本還不到約 2 年（`summary.train_weeks` 有目前週數）。

## 3. 補充來源（選用）

- 台股個股的董監持股、質押、重大訊息：若有連接 ToAlpha 公開資訊觀測站連接器，用 `resolve_stock` 後查 `insider_holdings`、`insider_trades`、`material_news`。
- 美股的新聞與情緒：Bigdata.com 連接器。
- 這些補充只做敘述，不要改寫 `dist` 裡的分數。

## 4. 更新資料

- **同步儀表板**：用 `list_triggers` 找名為「籌碼雷達：同步儀表板」的排程工作，用 `fire_trigger` 觸發。
- **重新抓資料並重算**：觸發 runner repo 的 GitHub Actions（先用 `add_repo` 加入 `chip-radar-runner`，access `push`）：
  ```bash
  gh api -X POST repos/esuper007/chip-radar-runner/actions/workflows/market-daily.yml/dispatches -f ref=main -f 'inputs[market]=tw'
  ```
  五個市場一次更新：改送 `all-markets.yml`（只要 `-f ref=main`，依序跑台、日、港、陸、美股）。同一時間只會有一個在執行、一個在排隊；查進度用 `GET .../actions/workflows/<檔名>/runs?per_page=5`。
  抓資料時若某天只公布了一部分（例如上櫃有、上市還沒出），筆數明顯偏少的近期日期會在下次執行自動重抓。
- **回補歷史**：`backfill.yml`，輸入 `market`、`start`（YYYY-MM-DD），會自動分段接力。
- **重新訓練模型與回測**：`weekly-model.yml`（每週六自動執行；平日只載入存好的模型）。

## 5. 修改指標或模型

- 各市場因子在 `pipeline/markets/<市場>.py` 的 `build_factors()`，評分面向與權重在同一函式的 `spec`、`weights`。
- 預測方法在 `pipeline/predict.py`；回測一律保留「訓練資料的標籤期間必須在測試期間開始前結束」的規則，避免偷看未來。
- 程式改在 `chip-radar-runner` 的 main 分支。這個 repo 和它的執行紀錄是公開的：程式只能記錄欄位名稱與筆數，不要印出資料數值、email 或權杖。改完先用 `tests/synthetic_tw.py` 產生假資料跑 `python -m pipeline.run build tw --backtest` 確認不會壞，再 push。
