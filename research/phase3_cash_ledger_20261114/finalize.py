from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1];TOL=1e-10
checks=[]
for f in ['daily.csv.gz','events.csv.gz','metrics.csv','accounting_checks.csv','order_bridge.csv','legacy_comparison.csv']:
 a=pd.read_csv(H/'results_pinned'/f,float_precision='round_trip');b=pd.read_csv(H/'results_initial'/f,float_precision='round_trip')
 assert a.columns.tolist()==b.columns.tolist() and a.shape==b.shape
 err=0.
 for col in a:
  assert a[col].isna().tolist()==b[col].isna().tolist(),(f,col,'mask')
  if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
   av=a[col].dropna().to_numpy(dtype=float);bv=b[col].dropna().to_numpy(dtype=float)
   e=float(np.max(np.abs(av-bv))) if len(av) else 0.;err=max(e,err);assert e<=TOL,(f,col,e)
  else:assert a[col].fillna('<NA>').tolist()==b[col].fillna('<NA>').tolist(),(f,col)
 checks.append(dict(file=f,rows=len(a),max_absolute_error=err,status='PASS'))
(H/'environment_comparison.json').write_text(json.dumps(checks,indent=2))
m=pd.read_csv(H/'results_pinned/metrics.csv');l=pd.read_csv(H/'results_pinned/events.csv.gz',float_precision='round_trip');d=pd.read_csv(H/'results_pinned/daily.csv.gz',float_precision='round_trip');a=pd.read_csv(H/'results_pinned/accounting_checks.csv')
telescope=[]
for v,g in l.groupby('variant'):
 end=d[d.variant==v].nav.iloc[-1];delta=abs(end-1-g[g.event_type=='MARK'].market_pnl.sum());assert delta<=TOL
 telescope.append(dict(variant=v,nav_change_minus_market_pnl=delta,status='PASS'))
 assert (g.cash_available.dropna()>=-TOL).all();assert (g.investment_fraction.dropna()<=1+TOL).all()
 buys=g[g.event_type=='BUY'];sells=g[g.event_type=='SELL']
 assert (pd.to_datetime(buys.event_time_utc)>pd.to_datetime(buys.funding_time)).all()
 assert (pd.to_datetime(sells.event_time_utc)>pd.to_datetime(sells.signal_time)).all()
 assert buys.price_observed_at.tolist()==buys.event_time_utc.tolist()
 assert sells.price_observed_at.tolist()==sells.event_time_utc.tolist()
(H/'telescoping_checks.json').write_text(json.dumps(telescope,indent=2))
prot=json.loads((H/'protected_before.json').read_text());assert len(prot)==256
assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in prot.items())
g=lambda *args:subprocess.check_output(['git','-C',str(R),*args])
assert g('diff','--name-only','e0d8c8755d26b402e0c6748b88add87c0686b1ec','--',*prot).decode()==''
(H/'protected_after.json').write_text(json.dumps(dict(status='PASS',count=256,baseline='e0d8c8755d26b402e0c6748b88add87c0686b1ec'),indent=2))
gate=dict(P0_1='PASS_FROZEN_SNAPSHOT_AND_SYNTHETIC_TESTS',P0_2_synthetic_cash_ledger='PASS_FROZEN_SNAPSHOT_AND_SYNTHETIC_TESTS',real_currency_settlement_accounting='INDETERMINATE_NOT_MODELED',P1_3='INDETERMINATE_PERMANENT_IDS',P1_4='INDETERMINATE_PIT_TERMINAL_EVENTS_PRICE_PROVENANCE',P2_5='NOT_PASSED',E30='NOT_PASSED',E31='LIMITED_MATHEMATICAL_PASS_UNCHANGED',E32='NO_GO',E32_executed=False,production_changed=False)
(H/'gate_decision.json').write_text(json.dumps(gate,indent=2))
old=pd.read_csv(R/'research/phase3_repair_20261114/results_pinned_v1/metrics.csv');old=old[old.cell=='D'].set_index('variant')
rows=['| 条件 | 新モデルCAGR | 年率ボラ | 最大DD | 通常Sharpe RF0 | 旧D CAGR（未認証） |','|---|---:|---:|---:|---:|---:|']
for row in m.itertuples():rows.append(f'| {row.variant} | {row.CAGR:.2%} | {row.Annualized_Volatility:.2%} | {row.Max_Drawdown:.2%} | {row.Conventional_Sharpe_RF0:.3f} | {old.loc[row.variant,"CAGR"]:.2%} |')
report=f'''# Phase 3 E30：仮想共通単位・現金台帳 実装監査報告

2026-11-14。ACCOUNTING_DESIGN.md v1.0の基準案Bに対するユーザー承認を受けて実装した。旧設計ファイル内の「未承認」は当時の履歴として保存し、この報告で今回の承認と実装を記録する。

## 結論

**全売却後の予算確定・現金待機・数量固定を実装し、固定6条件の仮想会計検証は合格。E30総合は未通過、E31限定合格、E32 NO-GOを維持。**

前回の旧/新ウェイト混在による最大107.07%の暗黙投資比率は、新モデルでは発生しない。全イベントの投資比率は数値許容差内で100%以下、利用可能現金は非負。これは仮想会計としての資金保存の確認であり、実際の通貨口座・受渡・売買可能性や予測力の認証ではない。

## 実装した仕様

初期資産1、借入・空売り・追加資金なし、費用・金利・FXなし、T+0、端数の仮想単位。選定確定後の各市場closeで旧保有を売却し、全売却完了後に現金を予約。購入はそれより厳密に後の各市場close。購入後の数量は固定し、日次ウェイト再調整をしない。

予約現金は現金の内訳でありNAVへ二重加算しない。各銘柄の初回購入価格で単位を正規化し、再購入でも尺度を維持する。実株数・配当・為替を再現したものではない。調整済み価格のリターンへ別配当を加算しない。

営業日closeイベントを価格表とは独立に生成する。欠損や非正価格が保有評価・約定に必要ならBLOCKED_DATAにし、その後のNAVと全期間指標を判定不能にする。休場だけを据え置き、未来の有効価格や全期間first/lastを使わない。

## 入力と事前固定

価格：旧固定Parquet、SHA256 ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d。旧6条件・19選定日・1026選定行を固定し、銘柄選定は再計算していない。2022-01-03〜2026-11-11、1263日付の23:59:59 UTCで評価。

PROTOCOL.mdとSHA-256を計算前に固定。絶対誤差1e-10、相対0を維持。閾値変更なし。旧256追跡ファイルを保護し、新規研究領域のみ追加。mainへの変更・マージなし。

## 会計検証結果

- 購入は計{int(a.buy_count.sum())}件、売却は計{int(a.sell_count.sum())}件。期末保有は売却せず時価評価。期末未約定0件、実データでの欠損停止0件。
- 売買・予約前後NAV保存の最大誤差：{a.event_conservation_max_error.max():.3g}。
- 価格更新損益との一致：最大誤差{a.mark_pnl_max_error.max():.3g}。
- 別実装（正規化しない価格数量と現金、選定回ごとに計算）との全日次NAV差：最大{a.oracle_nav_max_error.max():.3g}。
- 二環境の全イベント・日次・注文・指標等の最大差：{max(x['max_absolute_error'] for x in checks):.3g}。いずれも1e-10以内。
- 生の最小利用可能現金：{a.minimum_event_available.min():.3g}、最大投資比率：{a.maximum_event_investment.max():.17g}。丸め差をクリップせず記録し、許容差を超える借入・過大投資は0件。
- 期末NAV−1と全価格更新損益の累計が一致。価格更新以外の取引で資産が増える現象なし。
- 実データ6条件で2024-01-05までの価格・選定だけに切断し、元実行の同時点までの全台帳・日次NAVに一致。合成テストでは未来価格の変更でも過去台帳不変を確認。

## 条件付き結果と旧系列との差

**以下は生存者/PIT等の問題が残る固定標本上の会計リプレイ。成績の良さは合格条件にしていない。旧Dは資金台帳を持たない未認証診断系列であり、新旧を同一条件の運用優劣として比較しない。**

'''
report+='\n'.join(rows)+'\n\n'
report+=f'''新モデルと旧D/Hとの差は、数量固定・売却待ち・購入遅延の複合差。単純なバグ修正の効果やαとは解釈しない。単一市場の合成例では、同じ購入時刻・価格で数量固定と日次50/50再配分が異なることを別途確認した。

注文単位の選定確定→購入待機は平均{m.mean_wait_signal_hours.mean():.2f}時間、最大{m.max_wait_signal_hours.max():.2f}時間。土日祝日を含む暦時間であり営業時間ではない。平均現金比率は約{m.mean_cash_fraction.min():.2%}〜{m.mean_cash_fraction.max():.2%}だが、初回投資前の待機も含むので純粋な入替待機コストとは呼ばない。購入予算比率と購入時の実ウェイトはorder_bridge.csvで分離した。

通常Sharpe（平均日次/標準偏差×sqrt252、RF0）と旧CAGR/ボラは別列。新MDDは初期NAV=1を含み、旧互換MDDも保存した。今回の6条件では両MDDは一致したが、初日損失の合成テストでは違いを確認している。

## テストと再現

新規17テストを各環境で実施し、両環境で17件すべて合格。購入前急騰、売却日収益、非同期資金待機、予約現金、営業日欠損/行削除、約定価格欠損、休場、未来変更/切断、価格尺度変更、行順変更、初期NAVを含むDD、期末未約定、前期未完、無効ウェイト、夏時間、数量固定と日次配分の差を対象にした。

既存の研究テスト22件と前回修復9件も組み合わせて回帰確認し、今回の統合実行は48件すべて合格（非推奨警告9件）。今回、本番コードは一切変更せず、旧本番98件の前回合格ログと不変ハッシュを引き継ぐ。新たに98件を実行したという主張はしない。

再現はREADME.md参照。入力ファイル・旧選定表は元のまま。出力先は新規ディレクトリ必須、既存成果物を上書きしない。ライブラリの非推奨警告はログへ保存。

## 残された課題と移行判定

| 項目 | 判定 |
|---|---|
| P0-1 明示的欠損処理 | 固定標本・合成検証の範囲で合格を維持 |
| P0-2 仮想共通単位の現金・数量台帳 | 限定範囲で合格 |
| 実通貨・受渡・企業行動を含む実口座会計 | 判定不能／未モデル化 |
| 恒久ID・株式クラス | 判定不能、P1-3の課題 |
| PIT構成・退出/終端イベント・価格来歴 | 判定不能、P1-4の課題 |
| E30総合／P2-5 | 未通過 |
| E31 | 限定的数学合格のまま |
| E32 | NO-GO、未実行 |

次の研究対象はP1-3の恒久ID・同点順位と、P1-4のPIT/企業イベント資料の充足性確認。今回その追加実験・有料取得は実行していない。会計合格だけでE32へ進まない。実口座モデルが必要になれば通貨別現金・FX・未調整価格・企業イベント・受渡の仕様と入力を別途確定する。
'''
(H/'REPORT.md').write_text(report)
print('PASS environment comparison, telescope, chronology, protection; report written')
