# db_pipeline — DB → versioned dataset

從 annotation_web 的 SQLite live DB (`data/annotation/<dataset>/annotations.db`) 一次產出帶 manifest 的版本化資料集。**取代** legacy 鏈：`anotation2coco.py + merge_coco.py + process_coco.py`。

## 用法

從 repo root 執行：

```bash
$PY -m dataset_pipeline.db_pipeline.build
```

預設值已經涵蓋目前的目錄配置；常用旗標：

```bash
$PY -m dataset_pipeline.db_pipeline.build \
    --output data/processed \                 # 預設
    --annotation-root data/annotation \       # 預設
    --symptoms-json data/annotation/symptoms.json \  # 預設
    --healthy-root data/healthy_images \      # 預設；負樣本 pool（每個子資料夾＝一個 pseudo-dataset）
    [--labels-txt data/raw/labels.txt] \      # optional override
    --split-ratios 8 1 1 \                    # 預設
    --version-tag v_paper_camera_ready \      # 預設用 YYYY-MM-DD_HHMM
    [--strict-categories] \
    [--require-submit | --require-expert-submit]
```

預設不讀 `labels.txt`：`detection/` view 會排除 `symptoms.json` 內 `id=0` 的類別，並把其他所有類別合併成單一 `ABNORMAL`（COCO `category_id=0`）。只有需要自訂 detection 類別空間時才傳 `--labels-txt`。

## 輸出結構

```
data/processed/
  image_registry.json              ← 全域、跨版本持久；source of truth for ID + split
  current → 2026-05-17_1430        ← 永遠指向最新一次成功 build
  2026-05-17_1430/
    MANIFEST.json                  ← 版本元資料 + filter stats
    symptoms.json                  ← snapshot
    labels.txt                     ← 只有傳入 --labels-txt 時才會 snapshot
    image_index.json               ← 本版本實際包含的 image ID（依 split 分組）
    category_diff.txt              ← 有用到任何類別就寫出：已用類別+數量 / 定義未用 / 未知類別+數量
    full/{train,valid,test}/
      <id>.jpg                     ← 正樣本實體檔；負樣本為 symlink 到 ../../healthy_images/<split>/<id>.jpg
      _annotations.coco.json       ← symptoms.json 完整類別空間（含負樣本 image entry，isHealthy:true、0 annotation）
    detection/{train,valid,test}/
      <id>.jpg                     ← symlink 到 full（正樣本）或 healthy_images（負樣本）
      _annotations.coco.json       ← 預設排除 id=0，其餘全併為 ABNORMAL；或依 --labels-txt 重編號（含負樣本 image entry）
    healthy_images/{train,valid,test}/
      <id>.jpg                     ← 負樣本唯一實體檔；來源見下方 Filter 規則
```

## Filter 規則（順序）

從 SQLite `tasks` 表逐筆判定：

1. `comments_count > 0` → drop task
2. `--require-submit`：`general_editor + expert_editor` 都空 → drop
3. `--require-expert-submit`：`expert_editor` 空 → drop
4. image 在 `<dataset>/images/` 與 `<dataset>/healthy_images/` 都找不到 → drop + 計入 manifest
5. Per-bbox：
   - label 為空 / `box_xyxy` 不合法 / `x2<=x1` 或 `y2<=y1` → drop bbox
   - label 不在 `symptoms.json` → drop bbox + 計數
6. `healthy_region` 被任何 wound bbox 包覆 → 移除該 healthy
7. 健康任務（detections 空 OR 全 `healthy_region`）→ **drop**（計入 manifest `dropped_by_healthy`）。view 只放正樣本。

## 負樣本（healthy_images）

健康 DB task 一律丟棄（見上方規則 7）。負樣本改成從**資料夾**收集：

1. 每個 dataset 的 `<dataset>/healthy_images/`（`dataset` 名當 registry key 的 dataset）
2. `--healthy-root`（預設 `data/healthy_images`）下每個子資料夾＝一個 pseudo-dataset（子資料夾名當 dataset），加上 root 底下的散檔

防呆：若某 `(dataset, filename)` 已被本次正樣本佔用就跳過。每張負樣本一樣經 `image_registry` 發 ID、依 `md5(filename)` 分 split，實體檔寫到 `healthy_images/<split>/<id>.jpg`，並在 `full/` 與 `detection/` 兩個 COCO json 各補一筆 `isHealthy:true`、0 annotation 的 image entry，full/detection 的同名檔為 symlink。

## `--strict-categories`

預設行為：未知 label 跳過該 bbox，task 仍寫出。

加上此旗標時：先掃完全部 task，若有任何 unknown label 就**中止 build**（不寫 view），但仍會寫 `MANIFEST.json` + `category_diff.txt` 方便分析。

## ID 與 split 分配

- 新圖：拿 `image_registry.json` 內 `next_id` 當 ID，依 `md5(original_filename)` 決定 split
- 既有圖：ID 與 split 都從 registry 取，**永遠不變**
- 新增資料集 / 新增圖只會 append，不會打亂歷史 split

如果想 reset registry，刪掉 `data/processed/image_registry.json` 即可（會重新發號，舊版本的 ID 對不上）。

## 下游使用

```python
# 直接讀 current symlink
DATA_ROOT = "data/processed/current/full"           # vl_classifier 想吃這個
DATA_ROOT = "data/processed/current/detection"      # RF-DETR / cause_inference 想吃這個
```

兩個 view 的圖在實體層只存一份（`full/` 是實檔，`detection/` 用 symlink）。
