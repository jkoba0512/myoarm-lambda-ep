# bioRxiv → Biological Cybernetics 投稿チェックリスト

(last updated: 2026-05-01 R8 自主監査 — vector/scalar 表記厳密化済 / bioRxiv 投稿 ready)

## 投稿経路 (確定)

1. **bioRxiv pre-print** (Neuroscience / Bioengineering)
   - endorsement 不要、投稿後 1–2 営業日でオンライン
   - DOI 付与あり、後の正式投稿に紐付け可能
2. **Biological Cybernetics (Springer)** へ正式投稿
   - hybrid OA: APC を払わなければ paywall (許容)
   - Feldman λ-EP の伝統的ホーム誌
   - スコープ: theoretical/computational neuroscience, control of biological movement

## bioRxiv 投稿に必要なファイル

### 必須

- [x] **Manuscript PDF** — `paper/manuscript.pdf` (~24 page A4, ~280 KB)
  - `bash paper/build.sh` で再生成可能
  - 本文 ~7900 words + Figure captions ~870 + References (18 entries inline)
  - Abstract 275 words
  - **Figure 1-5 inline embedded** at the captions section (bioRxiv style)
- [x] **Figures (separate files)** — `figures/fig{1,2,3,4,5}.pdf`
  - 全 5 枚、Springer 規格 183mm 幅、≤ 50 KB/枚
  - PNG 版 (`figures/fig{1,2,3,4,5}.png`) も存在
  - bioRxiv 投稿時は manuscript.pdf に埋め込み済なので separate upload は optional
- [x] **References list** — manuscript.md 末尾に inline (18 entries)
- [x] **LICENSE** — MIT (`LICENSE` in repository root)
- [x] **README quickstart** — Phase 1-6 (MyoSuite paper) 再現コマンドを `README.md` 冒頭に追加済

### 推奨 (可能なら同時投稿)

- [ ] **Supplementary material PDF** — Bio Cyb 投稿時に作成
  - cerebellar gain-sweep table (joint K∈{0.05,0.1,0.2,0.5}, λ K∈{0.1,0.5,1.0})
  - F3 synergy ablation, F6 γ-Ia, F7 Kd-proximity, F15 task-space VT 等の negative result 詳細
  - F17 PD-no-cereb baseline の per-seed table (本文 §2.6 で言及済)
- [x] **Code repository** — このリポジトリ全体
  - `src/myoarm/` + `scripts/experiment_myo_p15_*.py` + `scripts/figures/*.py`
  - `results/experiment_myo_p15/*.json` (raw per-seed metric tables、F13/F16/F17)
  - `results/myo_cfc_data*/cfc_model.pt` (trained CfC weights)

### bioRxiv フォームで入力する metadata

- **Title**: Decoupling smoothness, accuracy, and kinematic invariance in
  biological reach: an ablation study of an equilibrium-point controller
  in a 34-muscle arm model
- **Author**: Jun Kobayashi (sole author)
  - Affiliation: Department of Intelligent and Control Systems, Graduate School
    of Computer Science and Systems Engineering, Kyushu Institute of Technology,
    680-4 Kawazu, Iizuka, Fukuoka 820-8502, Japan
  - ORCID: 0009-0002-6318-2617
  - Corresponding email: kobayashi.jun184@m.kyutech.ac.jp
- **Subject area**: Neuroscience > Systems Neuroscience
  (alt: Bioengineering)
- **Keywords**: equilibrium-point hypothesis · virtual trajectory ·
  visuomotor feedback · spinal reflexes · musculoskeletal model ·
  reach control · MyoSuite · ablation study
- **Abstract**: paper/manuscript.md の Abstract セクション (275 words)
- **License**: CC-BY 4.0 推奨
- **Conflict of interest**: なし
- **Funding**: No external funding

## Biological Cybernetics 正式投稿に必要な追加作業

bioRxiv 投稿が成功した後に着手:

- [x] ~~**Equivalence test (TOST)**~~ — R4 で §2.7 + §3.1 に組込済 (margin ±20mm/±25mm/±5°、p_TOST = 3e-3 / 1.8e-4 / 1.1e-8)
- [ ] **TOST margin 選定理由を supplement に詳細化** (R5 査読指摘): 「±20mm = absolute residual の 1/5」だけだとやや恣意的。target distance normalised error や 50mm tolerance との関係も添える
- [ ] **Abstract を 200-250 words に再削減** (現 304 words → Bio Cyb 推奨 200 words)
- [ ] **Springer LaTeX template への移植**
  - Springer Nature が提供する `sn-jnl.cls` (or `svjour3.cls`) を取得
  - `paper/manuscript.tex` (現状 pandoc 生成) を template に組み込み
  - bibliography style を Springer 指定 (Basic / Numeric / Author-Year) に変換
  - 現在の plain-text "References" を `\bibliography{references.bib}` + `bibtex` に置換
- [ ] **Cover letter** — 約 1 page
  - Why this journal (Feldman λ-EP の伝統的ホーム誌)
  - Why now (MyoSuite の登場で初めて 34-muscle で λ-EP を ablate できる)
  - Suggested reviewers ※ 検討
- [ ] **Author affiliations + ORCID**
- [ ] **Figure files** — Springer の解像度規定 (300 dpi 以上 / EPS or TIFF) に合わせ再 export
- [ ] **Highlights** (3–5 bullet) — もし誌指定があれば追加
- [ ] **Supplementary material PDF** 作成
  - Cerebellar gain sweep table (joint K∈{0.05,0.1,0.2,0.5}, λ K∈{0.1,0.5,1.0})
  - F3 synergy / F6 γ-Ia / F7 Kd-proximity / F15 task-space VT の negative-result 詳細
  - F17 PD-no-cereb baseline の per-seed table
  - tolerance 別 solve rate (50/75/100/150 mm) と target-distance-normalised error
  - approach-phase straightness vs full-window straightness

## Codex 査読対応履歴

### R1 (2026-04-30、第1回査読)

- ✅ baseline / ablation reference の混同 解消 (§2.6, §3.2)
- ✅ "equivalent to endpoint-PD" 誤記 解消 (§3.2)
- ✅ supplement 参照 → "released with code" に変更 (§2.5, §3.3)
- ✅ Fig 5 caption の cerebellum 記述 修正
- ✅ human-like overclaim Abstract / §1.3 / §5 で弱め
- ✅ Table 1 SD 補完 (direction error / straightness)
- ✅ paired Wilcoxon 追加 (§3.1, §3.2)
- ✅ MyoSuite/MuJoCo/Gymnasium バージョン明記 (§2.1)

### R2 (2026-04-30、第2回査読)

- ✅ #1 endpoint-PD baseline の cerebellar correction 片側問題: F17 追加実験 (no-cereb PD baseline n=50) を実行、§2.6 に「cerebellar branch is no-op」を統計的に明示
- ✅ #2/#3 Welch primary 問題: paired Wilcoxon を **primary** に格上げ、Welch を effect-size 補助に降格 (§2.7, §3.1)
- ✅ #4 §4.1 overclaim: 見出しを "Bell-shape and smoothness invariants emerge…" に変更、"by construction" → "under this controller architecture"
- ✅ #5 LICENSE/README/release claim: MIT LICENSE 追加、README に Phase 1-6 quickstart 追加、本文を "will be released with the pre-print" に弱め
- ✅ Minor: Fig 1 caption "Endpoint accuracy is preserved" → "Minimum-tip-error distributions overlap"; Fig 2b/2c に panel-level n を明示; Abstract 282→275 words

### R3 (2026-04-30、第3回査読)

- ✅ #1 §2.6 F17 wording: "statistically indistinguishable" → "practically negligible"
- ✅ #2 figure scripts を paired Wilcoxon 化: Fig 1/Fig 4 を再生成、min_err や reflex→accuracy の小コストが star 表示に
- ✅ #3 Conclusion + §1.3 wording: "statistically indistinguishable" → "practically equivalent"
- ✅ #4 README requirements.txt → `pip install -e .`
- ✅ #5 release tense 統一: §3.3 "is released" → "will be released"
- ✅ Table 1 / Fig 1 / Fig 4 caption の "Welch's t-test" → "paired Wilcoxon signed-rank test"

### R4 (2026-05-01、第4回査読)

- ✅ #1 TOST equivalence test 追加: margin ±20mm/±25mm/±5° で全 3 metric statistical equivalence 確立 (p_TOST ≤ 0.003)
- ✅ #2 baseline naming 統一: Abstract/§1.3/§2.6/§3.1/Table 1 で "endpoint-PD + spinal"
- ✅ #3 Table 1 d/p 分離: 別カラム + caption 注記
- ✅ #4 §3.1 数値順序: 主語と数値の順序一致
- ✅ #5 solve rate 防御的表現削除: "not driven by the controller" 撤回
- ✅ #6 human-like overclaim narrowing: §3.1 見出しを "human-range peak timing"、Bell-shape paragraph に skewness/transient 併記
- ✅ #7 Fig 4 caption Reference 強調: bold "Reference = pure λ + visuomotor, not endpoint-PD"
- ✅ #8 figures 余分ファイル削除: `figures/　.png`
- ✅ #9 §2.6 F17 disclosure 圧縮 (7段→2段)
- ✅ #10 mechanistic-ablation framing: Abstract と §5 末尾に追加
- ✅ #11 References DOI 訂正: Feldman & Levin 1998→1995, Sarlegna & Sainburg 2014→2009, Caggiano 2023→2022 vol 168, Lakens 2017 追加, DEP-RL を Schumacher et al. 2023 に修正, 全 inline ref に DOI 付与

### R5 (2026-05-01、第5回査読 — 表記整合化のみ)

- ✅ #1 §3.1 TOST 表現を統計的に正確に: "rejects inferiority at both bounds" → "rejects both equivalence-null tests"
- ✅ #2 "pre-specified" → "pre-defined" (post hoc 疑念回避、全箇所)
- ✅ #3 §4.1 / §5 / §4.4 の "endpoint-PD baseline" → "endpoint-PD + spinal baseline" 統一
- ✅ #4 Fig 1 / Fig 2 caption の旧 overclaim 修正 ("statistically indistinguishable" 削除、"human-like bell" → "peak timing inside human band")
- ✅ #5 Fig 1 caption の "statistically indistinguishable" を "remain within the pre-defined ±20 mm practical equivalence margin" に
- ✅ #6 §3.1 で direction error TOST 結果 (p≈1.1×10⁻⁸) と final tip error TOST 結果 (p≈1.8×10⁻⁴) を追記
- ✅ #7 §3.2 冒頭に lower-powered ablation の note を追加
- ✅ Minor: Abstract 末尾 caveat を short 化 (Codex 提案文採用) → 304 words
- ✅ §4.1 "are what produces" → "can emerge in this architecture when"

### R6 (2026-05-01、第6回査読 — 表現/数値整合の最終仕上げ)

- ✅ #1 §3.1 direction error 重複文を統合 (1 文に圧縮)
- ✅ #2 Fig 1 caption の数値誤り修正: "All four ... 90–110 mm" は λ-traj no-visuo (148mm) 含むので不正確 → "headline conditions cluster near ≈100 mm; the no-visuomotor λ-trajectory condition reaches less accurately (≈148 mm)"
- ✅ #3 §2.7 残存 "pre-specify" 2 箇所 ("we pre-specify" / "We do not pre-specify") を "define" / "do not define" に置換
- ✅ #4 §2.7 + Fig 2a caption の "endpoint-PD baseline" → "endpoint-PD + spinal baseline" 統一
- ✅ #5 Fig 1 図中ラベル "Endpoint PD\n(engineering)" → "Endpoint PD\n+ spinal" + 再生成 (本文・caption・図中ラベル一致)
- ✅ Audit で発見した R5/R6 narrowing 取り溢し 7 件補完: §1.3 #4 cerebellum caveat / §1.3 #3 axes / §3.2 reflex subheading + wrap-up + 結論 / §4.2 VT secondary / §4.5 implications

### R7 (2026-05-01、第7回査読 — bioRxiv 投稿 ready 判定)

- ✅ Codex 判定: **「bioRxiv 投稿前原稿として ready と見てよい」「論文本体の追加修正は必須ではない」**
- ✅ #1 Fig 4 caption の "not the endpoint-PD baseline of §3.1" → "not the endpoint-PD + spinal baseline of §3.1" (徹底感のため optional 修正)
- ✅ #2 Fig 1 caption (a) "Endpoint-PD (gray)" → "Endpoint-PD + spinal (gray)" (徹底感のため optional 修正)
- ✅ Minor: SUBMISSION-CHECKLIST に R6/R7 履歴追記
- ⏸ Abstract 304 words: bioRxiv は OK、Bio Cyb 投稿時に 200-250w 圧縮 (公式規定 150-250 words 確認済)

### R8 (2026-05-01、自主監査 — vector/scalar 表記の厳密化)

- ✅ §2.2 末尾に **Notation 段落**を追加: $\mathbf{1}_{n} \in \mathbb{R}^{n}$ を all-ones vector として定義 (指示関数 $\mathbf{1}[\cdot]$ と区別)、scalar±vector の broadcasting と $\mathrm{clip}/\max$ on vector の componentwise 規約を明示
- ✅ Eq. (2): $\mathbf{L}_{\rm target} - \lambda_{0}$ → $\mathbf{L}_{\rm target} - \lambda_{0}\mathbf{1}_{34}$
- ✅ §2.2 prose: $\boldsymbol{\lambda}_{s} = \mathbf{L}(\mathbf{q}_{0}) - \lambda_{0}$ → $\dots - \lambda_{0}\mathbf{1}_{34}$
- ✅ Eq. (4): visuomotor update も $\lambda_{0}\mathbf{1}_{34}$ 化
- ✅ §2.6 Eq.: $\mathbf{a} = \mathrm{clip}(\dots + a_{\rm bias}, 0, 1)$ → $\dots + a_{\rm bias}\mathbf{1}_{34}, 0, 1)$
- ✅ Fig 5 caption: $\boldsymbol{\lambda}_{\rm target} = \mathbf{L}(\mathbf{q}^{*}) - \lambda_{0}$ → $\dots - \lambda_{0}\mathbf{1}_{34}$
- ✅ Fig 5 caption の $\max(\mathbf{L}-\boldsymbol{\lambda}_{\rm eff}, 0)$ と $\mathrm{clip}(\cdot,0,1)$ on vec は新規 Notation 規約でカバー
- 効果: Bio Cyb / mathematical neuroscience 査読者の "Eq. (2): notation ambiguity / dimension mismatch" 指摘を未然に防止
- 影響: 本文 6 行修正 + Notation 段落 1 段追加。意味・実装は変更なし、表記の厳密化のみ

### R8e (2026-05-01、自主監査 — activation vector 表記の bold 統一)

- 🐛 発見: `\mathbf{a}^{\rm efcopy}` (line 358) と `\mathbf{a}` (line 399) が既に bold だったが、§2.4 末尾の `\Delta a^{\rm cereb}`、§2.5 の `\Delta a^{\rm cereb} = J_{\rm act}^+ \boldsymbol{\tau}_{\rm cereb}`、Fig 5 caption の `a_{\rm base}`/`\Delta a_{\rm Ia,Ib,RI}`/`a_{\rm total}` が細字のままで内部不整合
- ✅ vector form (添字なし) を 7 箇所すべて bold 化:
  - line 347: `\Delta a^{\rm cereb}` → `\Delta\mathbf{a}^{\rm cereb}`
  - line 365: `\Delta a^{\rm cereb} = ...` → `\Delta\mathbf{a}^{\rm cereb} = ...`
  - line 1110: `a_{\rm base} = clip(...)` → `\mathbf{a}_{\rm base} = clip(...)`
  - line 1111-1114: `\Delta a_{\rm Ia/Ib/RI}` → `\Delta\mathbf{a}_{\rm Ia/Ib/RI}`
  - line 1115: `a_{\rm total}` → `\mathbf{a}_{\rm total}`
- Eq. (1) `a^{\rm base}_i` / Eq. (5) `a^{\rm total}_i` 等の **indexed form は細字維持** (scalar component として正しい)
- Fig 5 内図 (R8d で `\mathbf{a}_{\rm total}` 化済) と完全整合
- 効果: vec/scalar 表記が manuscript 全体で統一され、reviewer の "notation inconsistency" 指摘を未然に防止

### R8d (2026-05-01、自主監査 — Fig 5 数式を manuscript と整合)

- 🐛 発見: `scripts/figures/fig5_architecture.py` の 8 箇所の数式が R8/R8b 修正後の manuscript 表記と不整合。`\lambda` (細字)、`L` (細字)、broadcasting `\mathbf{1}_{34}` 欠落、clip 引数欠落
- ✅ matplotlib mathtext bold Greek 描画テストで `dejavusans` (デフォルト fontset) なら `\boldsymbol{\lambda}` が太字 𝝀 として描画されることを検証 (cm/stix/stixsans は失敗)
- ✅ 8 箇所一括修正:
  - IK 箱: `\lambda_{\rm target} = L(\mathbf{q}^*) - \lambda_0` → `\boldsymbol{\lambda}_{\rm target} = \mathbf{L}(\mathbf{q}^*) - \lambda_0\mathbf{1}_{34}`
  - virtual traj 箱: 全 `\lambda` を `\boldsymbol{\lambda}` に
  - arrow label 4 箇所 (λ_target / Δλ_target / λ(t) / λ_eff) を bold 化
  - Δλ_cereb 箱を bold 化
  - α-MN 箱: `c\cdot\max(L-\lambda_{\rm eff}, 0)` → `c_{\lambda}\cdot\max(\mathbf{L}-\boldsymbol{\lambda}_{\rm eff}, 0)`、clip 引数 (0, 1) も追加
- 効果: Fig 5 が manuscript の Eq.(2)/(4)/Fig 5 caption と完全整合。reviewer が figure と本文を行き来しても表記の一貫性が保たれる

### R8c (2026-05-01、自主監査 — 本文中の file format 言及を除去)

- 🐛 発見: §2.6 末尾 (line 416) に "full table in the released results JSON" の表現あり。Bio Cyb / Springer 系では本文に file format ("JSON", "CSV") を書くのは informal で慣習外
- ✅ Fix: "full table in the released results JSON" → "complete metric × test table provided in the released supplementary results"
- 効果: line 510 ("...will be released with the pre-print")、line 752 ("...full sweep table will be released") と表現が整合。Bio Cyb 投稿時の "Supplementary Table SX" への書き換えもスムーズに
- 影響: 本文 1 箇所のみ。意味は不変、慣習表現に整合化

### R8b (2026-05-01、自主監査 — 太字 Greek レンダリング bug fix)

- 🐛 発見: 旧 PDF で $\boldsymbol{\lambda}_{\rm target}$, $\boldsymbol{\tau}_{\rm cereb}$ 等の Greek vector が **regular italic で出力**されていた (太字にならない bug)。原因: pandoc preamble で `amssymb` が `unicode-math` より先に load され、`\boldsymbol` が再定義されないまま amssymb 版にとどまる
- ✅ Fix: `paper/preamble.tex` 新設 + `build.sh` に `--include-in-header=preamble.tex` 追加。`\AtBeginDocument{\let\boldsymbol\symbf}` で `\boldsymbol` を unicode-math 版に強制
- 結果: $\boldsymbol{\lambda}, \boldsymbol{\tau}, \Delta\boldsymbol{\lambda}, \boldsymbol{\lambda}(t)$ 等 38 箇所すべて bold italic 𝝀, 𝝉, ... で正しくレンダリング。スカラー $\lambda_0, \tau, \mu, \gamma$ と視覚的に明確に区別
- 影響: manuscript.md 編集ゼロ、preamble.tex (4 lines) + build.sh (1 line) のみ

## Pre-submission チェックリスト

### 内容

- [x] Abstract が 250–310 words 以内 (現 304 words ✓ for bioRxiv; Bio Cyb 用に 200-250 へ削減残)
- [x] Title/Abstract に出てくる主要数値 (n=50, d=-7.39, vpr=0.40-0.50) が本文と整合
- [x] §1–§5 の見出し階層が一貫
- [x] 全 figure が本文中で参照されている (Fig 1: §3.1; Fig 2: §3.1; Fig 3: §3.1; Fig 4: §3.2; Fig 5: §1.3)
- [x] 全引用文献が References セクションに存在 (20 entries, cited 全て in-list、全 entry に DOI/arXiv/URL 付与)
- [x] TOST equivalence margins (±20mm/±25mm/±5°) が §2.7 で pre-defined、§3.1 で 3 metric 全部 reject 確認
- [x] 単位表記が `m s^{-1}` で統一
- [x] λ-EP / VT / visuomotor / reflexes の符号付き Cohen's d が本文と Fig 4 caption で整合
- [x] paired Wilcoxon が primary、Welch が confirmatory として一貫
- [x] §2.6 baseline 説明と F17 追加実験結果が整合

### 技術

- [x] PDF が xelatex で完走 (`bash paper/build.sh`)
- [x] Unicode 文字 (λ, γ, μ, ±, ≤, ≈) が PDF で正しくレンダリング
- [x] Equations (1)–(6) が番号付きで参照可能
- [x] Table 1 の縦線 / 罫線が読めるレイアウト

### コード公開

- [x] LICENSE ファイル (MIT) — `/LICENSE`
- [x] README に Quickstart 追加済 (`README.md` 冒頭)
- [ ] DOI 取得 (Zenodo 連携) — bioRxiv 投稿時に code DOI も付与しておく
- [x] `src/myoarm/env_utils.py:deterministic_reset` を独立して docstring 化済

## 投稿当日の手順

```bash
# 1. PDF 再生成して中身確認
bash paper/build.sh
xdg-open paper/manuscript.pdf       # 目視チェック

# 2. figures が最新か確認
ls -la figures/

# 3. tarball 作成 (オプション)
cd /home/jkoba/SynologyDrive/00-Research/neuro-arm-control
tar -czf /tmp/biorxiv-submission.tar.gz \
    paper/manuscript.pdf \
    figures/fig*.pdf

# 4. bioRxiv (https://www.biorxiv.org/submit) でフォーム記入 + アップロード
```

## 残課題 (緊急度順)

1. ~~**affiliation / funding** を決定~~ (2026-05-01 確定: KIT 飯塚 / no external funding)
2. **Zenodo DOI** 取得 (GitHub 連携で 30 分)
3. **manuscript PDF 再ビルド** (`bash paper/build.sh`) で author block を最終確認
4. **Bio Cyb 用追加作業** は bioRxiv 投稿後で良い (TOST / Springer template / Supplementary)
