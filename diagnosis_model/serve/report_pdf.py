"""固定模板式診斷報告 → PDF（正式病例報告版式）。

吃 serve `/diagnose` 回傳的報告 dict（含 base64 data-URI 圖），用 Jinja2 套固定
模板 + WeasyPrint 出 A4 PDF。**不重跑推論**——只把現成資料與圖排版成可存檔文件，
對應論文第四章「固定模板式結構化報告」。中文以系統內 Noto Sans/Serif CJK TC 呈現。

版式仿魚病中心病例報告：機構抬頭 + 外框基本資料表 + 分節 (一)(二)(三) + 圖N。
"""
from __future__ import annotations

from jinja2 import Template
from weasyprint import HTML

# 報告抬頭——依單位自行修改。
REPORT_ORG = "國立臺灣海洋大學　魚病智慧診斷輔助系統"
REPORT_TITLE = "AI 魚病診斷輔助報告"

# 報告區塊對應 _serialize() 之輸出 schema。處置建議／專家覆核屬論文未來工作，不列入 PDF。
_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 16mm 16mm; }
  * { font-family: "Noto Sans CJK TC", sans-serif; }
  body { color: #1a1a1a; font-size: 11px; line-height: 1.55; }
  .org { font-family: "Noto Serif CJK TC", serif; font-size: 19px; font-weight: 700;
         text-align: center; letter-spacing: 1px; }
  .title { font-family: "Noto Serif CJK TC", serif; font-size: 15px; font-weight: 700;
           text-align: center; margin: 2px 0 10px; }
  .muted { color: #6b7280; }
  .xs { font-size: 9.5px; }

  table.info { width: 100%; border-collapse: collapse; margin-bottom: 4px; }
  table.info td { border: 1px solid #555; padding: 4px 8px; vertical-align: middle; }
  table.info td.lab { background: #f1f1f1; font-weight: 600; text-align: center; width: 14%; white-space: nowrap; }

  h2 { font-size: 12.5px; font-weight: 700; margin: 14px 0 5px; }
  table.data { width: 100%; border-collapse: collapse; margin: 4px 0; }
  table.data th, table.data td { border: 1px solid #999; padding: 3px 6px; text-align: left; vertical-align: top; }
  table.data th { background: #f4f6f8; color: #374151; font-weight: 600; }
  td.num, th.num { text-align: right; }

  .figrow { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }
  .figrow .item { text-align: center; }
  .fig { max-width: 100%; border: 1px solid #999; }
  .fig.wide { max-height: 72mm; }
  .crop img { width: 80px; height: 80px; object-fit: cover; border: 1px solid #999; }
  .case img { width: 96px; height: 72px; object-fit: cover; border: 1px solid #999; }
  .cap { color: #6b7280; font-size: 9px; margin-top: 1px; }

  .green { border: 1px solid #34d399; background: #ecfdf5; color: #047857;
           padding: 8px 12px; border-radius: 4px; }
  .twocol { display: flex; gap: 10px; }
  .twocol > div { flex: 1 1 0; min-width: 0; text-align: center; }
  .foot { margin-top: 16px; padding-top: 6px; border-top: 1px solid #ccc; color: #6b7280; }
</style>
</head>
<body>
  {% set fig = namespace(n=0) %}
  <div class="org">{{ org }}</div>
  <div class="title">{{ title }}</div>

  <table class="info">
    <tr>
      <td class="lab">病例編號</td><td>{{ r.meta.case_id }}</td>
      <td class="lab">報告產生時間</td><td>{{ r.meta.timestamp.replace("T", " ") }}</td>
    </tr>
    <tr>
      <td class="lab">病灶數量</td><td>{{ r.n_lesions }}</td>
      <td class="lab">診斷資料版本</td><td>{{ r.meta.data_version or "—" }}{% if r.meta.delta_cases %}（含 {{ r.meta.delta_cases }} 筆即時新增案例）{% endif %}</td>
    </tr>
    <tr>
      <td class="lab">補充描述</td><td colspan="3">{{ r.meta.text or "未提供" }}</td>
    </tr>
    <tr>
      <td class="lab">判定結果</td>
      <td colspan="3">{{ "健康（未偵測到明顯異常）" if r.abstain else "疑似異常，進行病因分析" }}</td>
    </tr>
  </table>

  {% if r.abstain %}
  <h2>（一）判定結果</h2>
  <div class="green">系統未在魚體表面偵測到明顯異常，判定為健康，未進行病因分析。</div>
  <div class="figrow"><div class="item">
    <img class="fig wide" src="{{ r.boxes_image }}">
    <div class="cap">送檢影像</div>
  </div></div>
  {% else %}

  <h2>（一）病灶定位分析</h2>
  <div class="twocol">
    <div>
      {% set fig.n = fig.n + 1 %}
      <img class="fig wide" src="{{ r.boxes_image }}">
      <div class="cap">圖{{ fig.n }}　原圖與病灶定位框</div>
    </div>
    <div>
      {% set fig.n = fig.n + 1 %}
      <img class="fig wide" src="{{ r.heatmap }}">
      <div class="cap">圖{{ fig.n }}　異常區域熱力圖</div>
    </div>
  </div>
  <table class="data">
    <thead><tr><th>病灶</th><th>主要症狀</th><th class="num">異常程度</th><th>可能症狀（信心）</th></tr></thead>
    <tbody>
    {% for l in r.lesions %}
      <tr>
        <td>L{{ l.idx + 1 }}</td><td>{{ l.label_zh }}</td>
        <td class="num">{{ "%.2f"|format(l.det_score) }}</td>
        <td>{% for it in l.top_k %}{{ it.label_zh }} {{ "%.0f"|format(it.prob * 100) }}%{% if not loop.last %}　/　{% endif %}{% endfor %}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  <h2>（二）相似案例檢索</h2>
  <div class="figrow">
    {% for c in r.retrieved %}
      {% set fig.n = fig.n + 1 %}
      <div class="item case"><img src="{{ c.image }}">
        <div class="cap">圖{{ fig.n }}　#{{ c.rank }}　相似度 {{ "%.3f"|format(c.similarity) }}</div></div>
    {% endfor %}
  </div>

  <h2>（三）疑似病因排序</h2>
  <table class="data">
    <thead><tr><th>排序</th><th>疑似病因</th><th class="num">AI 評分</th><th class="num">相似案例數</th></tr></thead>
    <tbody>
    {% for c in r.causes %}
      <tr><td>#{{ c.rank }}</td><td>{{ c.text }}</td>
        <td class="num">{{ "%.3f"|format(c.score) }}</td>
        <td class="num">{{ c.support if c.support is not none else "—" }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
  {% if r.causes %}
  {% set fig.n = fig.n + 1 %}
  <div class="figrow"><div class="item" style="width:100%">
    <img class="fig" src="{{ r.causes_breakdown }}" style="width:100%">
    <div class="cap">圖{{ fig.n }}　各疑似病因的證據來源比例</div>
  </div></div>
  {% endif %}
  {% endif %}

  <div class="foot xs">
    本報告由 AI 模型輔助產生，僅供診斷參考，須經魚病診斷專家覆核。
  </div>
</body>
</html>
"""
)


def render_report_pdf(report: dict) -> bytes:
    """report dict（/diagnose 之輸出 schema）→ PDF bytes。"""
    html = _TEMPLATE.render(r=report, org=REPORT_ORG, title=REPORT_TITLE)
    return HTML(string=html).write_pdf()
