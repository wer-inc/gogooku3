# お知らせ

* <mark style="color:red;">2025.10.17 キャッシュ・フロー計算書を財務諸表に追加しました。詳細は</mark>[<mark style="color:red;">こちらのページ</mark>](https://jpx.gitbook.io/j-quants-ja/api-reference/statements-1#paramtabiresuponsu)<mark style="color:red;">のレスポンス例を参照ください。</mark>
* 2025.8.22 Standardプラン及びPremiumプランで取得可能な日々公表信用取引残高APIをリリースしました。
* 2025.7.18 配当金情報APIの提供データの更新タイミングが変更になりました。
* 2025.5.30 Standardプラン及びPremiumプランで取得可能な空売り残高報告APIをリリースしました。\
  2025.5.2 データ修正履歴を更新しました。
* 2025.1.27 上場銘柄一覧APIのAPI概要について追記しました。上場銘柄一覧API、株価四本値API、指数四本値API、TOPIX四本値API及び業種別空売り比率APIの提供データの更新タイミングが変更になりました。
* 2024.12.3 投資部門別情報APIの留意点に過誤訂正があった際のデータ更新タイミングについて追記しました。
* 2024.11.5 株価四本値API、指数四本値API、TOPIX四本値API及び業種別空売り比率APIの提供データの更新タイミングが変更になりました。
* 2024.9.20 データ修正履歴を更新しました。
* 2024.8.26 現時点で判明している制約事項を更新しました。
* 2024.8.20 決算発表予定日の留意事項を更新しました。
* 2024.8.16 Premiumプランで取得可能な先物、オプションの四本値を取得できる新規のAPIをリリースしました。
* 2024.8.2 データ修正履歴を更新しました。
* 2024.7.22 四半期開示見直し対応に伴い、財務情報APIの項目に"SignificantChangesInTheScopeOfConsolidation"（期中における連結範囲の重要な変更）が追加されます。詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/statements)を参照ください。

<details>

<summary>過去のお知らせ</summary>

* 2024.6.17 データ修正履歴を更新しました。
* 2024.3.28 指数四本値APIに業種別や市場別等の指数を追加しました。
* 2024.3.28 取引カレンダーAPIの[提供データの更新タイミング](https://jpx.gitbook.io/j-quants-ja/outline/data-update)が変更になりました。
* 2024.2.28 データ修正履歴を更新しました。
* 2023.12.20 各種指数四本値を取得できる新規のAPI(/indices)をStandardとPremiumプラン向けに追加いたしました。
* 2023.11.28 株価四本値のデータ項目概要におけるAfternoonTurnoverValueに関する記載漏れを修正しました。
* 2023.11.07 データ修正履歴を更新しました。
* 2023.10.27 APIからのレスポンスをGzip化するように変更しております。利用パターンによっては解凍処理が必要となる場合がございます。詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/attention#gzip)を参照ください。
* 2023.09.22 データ修正履歴を更新しました。
* 2023.08.29 財務諸表(BS/PL)の[プランごとに利用可能なAPIとデータ期間](https://jpx.gitbook.io/j-quants-ja/outline/data-spec)を訂正しました。
* 2023.08.28 Premiumプランで取得可能な決算短信の詳細な情報を取得できる新規のAPIをリリースしました。
* 2023.06.30 上場銘柄一覧に、StandardとPremiumプランで取得可能な貸借信用区分を追加しました。
* 2023.6.16 株価四本値にストップ高・ストップ安のフラグを追加しました。データ量の増加に伴いページング対応が必要となる場合があります。ページング方法は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/attention#resuponsunopjingunitsuite)を参照ください。
* 2023.6.12 今後のリリース予定を更新しました。6月16日にメンテナンス作業に伴う株価四本値APIの利用制限を予定しておりますのでご確認ください。
* 2023.6.9 現時点で判明している制約事項を更新しました。
* 2023.5.12 今後のリリース予定を更新しました。
* 2023.5.8 営業日カレンダーを取得できる新規のAPIをリリースしました。
* 2023.4.27 ページング機能をリリースしました。今後のリリース予定を更新しました。\
  また、5月10日にメンテナンス作業に伴うAPIの利用制限を予定しておりますので御確認ください。
* 2023.4.3 J-Quants API（有償版）を正式にリリースしました。

</details>

## リリースノート <a href="#release_note" id="release_note"></a>

### リリース済みのもの

{% tabs %}
{% tab title="最近のリリース情報" %}

| リリース日        | 変更内容                                          | 備考                                                                                                                                                                                                  |
| ------------ | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024年8月16日   | 先物、オプションの四本値を取得できる新規のAPIをリリースしました。            | 先物APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/futures)、オプションAPIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/options)を参照ください。                                |
| 2024年3月28日   | 指数四本値APIに業種別や市場別等の指数を追加しました。                  | APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/indices)を参照ください。                                                                                                                  |
| 2023年12月20日  | 各種指数の四本値を取得できる新規のAPIをリリースしました。                | APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/indices)を参照ください。                                                                                                                  |
| 2023年10月27日  | データ通信量削減を目的としてAPIからのレスポンスをGzip化するように変更しております。 | <p>本仕様変更に伴うクライアント側での対処は基本的に不要ですが、ケースによっては解凍処理が必要となります。</p><p>詳細は<a href="../api-reference/attention#gzip">こちらのページ</a>を参照ください。</p>                                                                   |
| 2023年8月28日   | 決算短信の詳細な情報を取得できる新規のAPIをリリースしました。              | APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/statements-1)を参照ください。                                                                                                             |
| 2023年6月30日   | 上場銘柄一覧に貸借信用区分を追加しました。                         | APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info)を参照ください。                                                                                                              |
| 2023年6月16日   | 株価四本値にストップ高・ストップ安のフラグを追加しました。                 | <p>APIの詳細は<a href="api-reference/daily_quotes">こちらのページ</a>を参照ください。<br>データ量の増加に伴いページング対応が必要となる場合があります。ページング方法は<a href="../api-reference/attention#resuponsunopjingunitsuite">こちらのページ</a>を参照ください。</p> |
| {% endtab %} |                                               |                                                                                                                                                                                                     |

{% tab title=" 過去のリリース情報" %}

| リリース日         | 変更内容                                   | 備考                                                                                                                                                                                                                   |
| ------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023年5月8日     | 営業日カレンダーを取得できる新規のAPIをリリースしました          | APIの詳細は[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/trading_calendar)を参照ください。                                                                                                                          |
| 2023年4月27日    | レスポンスのデータサイズが大きい場合にページングする機能をリリースしました。 | <p>・ページング方法の詳細は<a href="../api-reference/attention#resuponsunopjingunitsuite">こちらのページ</a>を参照ください<br>・本対応より以下の事象は解決しております。<br>- 財務情報APIにおいて、2022年5月13日のデータを日付指定で取得できない<br>- 投資部門別情報APIにおいて、日付を指定しないで全日のデータを取得できない</p> |
| 2023年4月3日     | 有償版を正式にリリースしました。                       |                                                                                                                                                                                                                      |
| {% endtab %}  |                                        |                                                                                                                                                                                                                      |
| {% endtabs %} |                                        |                                                                                                                                                                                                                      |

### データ修正履歴

{% tabs %}
{% tab title="最近の修正履歴" %}

<table><thead><tr><th width="165">修正日</th><th width="121">修正対象API</th><th width="103">対象期間</th><th>修正内容</th></tr></thead><tbody><tr><td>2024年9月20日</td><td>オプション四本値</td><td>2010年1月4日-2014年3月20日</td><td>指数オプションについて前場・後場四本値の値を修正</td></tr><tr><td>2024年9月20日</td><td>先物四本値</td><td>2010年以降</td><td>指数先物についてDaySession四本値の値を修正</td></tr><tr><td>2024年8月2日</td><td><p>財務情報</p><p>財務諸表</p></td><td>ー</td><td>詳細は<a href="correction-history#on-2024-08-02">こちらのページ</a>を参照ください。</td></tr><tr><td>2024年6月17日</td><td>上場銘柄一覧</td><td>ー</td><td>ScaleCategoryの値が一部誤って"-"になってしまっているのを修正</td></tr><tr><td>2024年2月28日</td><td><p>財務情報</p><p>財務諸表</p></td><td>2009年1月13日-2024年2月8日</td><td>詳細は<a href="correction-history#on-2024-02-28">こちらのページ</a>を参照ください。</td></tr><tr><td>2023年11月7日</td><td>業種別空売り比率</td><td>2023年11月6日</td><td>2023年11月6日のデータ全体を修正。</td></tr><tr><td>2023年9月22日</td><td>売買内訳データ</td><td>ー</td><td>一部日付に存在した実在しない以下銘柄コードのデータを削除<br>銘柄コード：20000, 30000, 50000</td></tr></tbody></table>
{% endtab %}

{% tab title="過去の修正履歴" %}

<table><thead><tr><th width="164">修正日</th><th width="124">修正対象API</th><th width="159">対象期間</th><th width="336">修正内容</th></tr></thead><tbody><tr><td>2023年4月10日</td><td>財務情報</td><td>2008年7月7日ー2014年3月31日</td><td><p>以下の項目にいて欠損データを修正</p><ul><li>ResultDividentPerShareAnual</li></ul><p>各項目について、前会計期間の値が入っててしまっている箇所があったため、それらを当会計期間の値へ修正</p></td></tr><tr><td>2023年4月10日</td><td>株価四本値</td><td>2023年3月28日</td><td>欠損していた2023年3月28日のデータを追加</td></tr><tr><td>2023年4月4日</td><td>財務情報</td><td>2008年7月7日ー2014年3月31日</td><td><p>以下の項目ついて欠損データを修正</p><ul><li>TypeOfCurrentPeriod</li><li>CurrentPeriodStartDate</li><li>CurrentPeriodEndDate</li><li>CurrentFiscalYearStartDate</li><li>CurrentFiscalYearEndDate</li></ul></td></tr><tr><td>2023年4月4日</td><td>オプション四本値</td><td>2008年5月7日ー2016年7月15日</td><td>Month（限月）について、YYYY-MM形式となるよう修正</td></tr></tbody></table>
{% endtab %}
{% endtabs %}

### 現時点で判明している制約事項

現時点で判明している制約事項や問題事象について記載しています

{% tabs %}
{% tab title="現在判明している制約事項" %}

<table><thead><tr><th width="108">追加日</th><th width="109">対象のAPI</th><th width="201">内容</th><th width="219">回避方法</th><th>解消日</th></tr></thead><tbody><tr><td>なし</td><td></td><td></td><td></td><td></td></tr></tbody></table>
{% endtab %}

{% tab title="解消済み" %}

<table><thead><tr><th width="108">追加日</th><th width="109">対象のAPI</th><th width="201">内容</th><th width="219">回避方法</th><th>解消日</th></tr></thead><tbody><tr><td>2024年8月26日</td><td>先物四本値</td><td>指数先物についてDaySessionの四本値の値が取得できない。</td><td></td><td>2024年9月20日</td></tr><tr><td>2023年6月9日</td><td>財務情報</td><td>TypeOfCurrentPeriodとCurrentPeriodEndDateに誤り<br>・銘柄コード36330の開示日付2017-04-27、2017-07-27、2017-10-30<br>・銘柄コード60260の開示日付2015-04-28、2015-07-29</td><td><p>TypeOfDocumentの値でどの期間かを確認することが可能です</p><p><a href="api-reference/statements/typeofdocument">開示書類種別</a></p></td><td>2024年2月28日</td></tr><tr><td>2023年4月10日</td><td>投資部門別情報</td><td>日付を指定しないで全日のデータを取得できない</td><td>日付やセクションを指定して、取得する対象を絞ってリクエストしてください。</td><td>2023年4月27日</td></tr><tr><td>2023年4月3日</td><td>財務情報</td><td>2022年5月13日のデータを日付指定で取得できない</td><td>日付に加えて銘柄コードを指定して、取得する対象を絞ってリクエストしてください。</td><td>2023年4月27日</td></tr></tbody></table>
{% endtab %}
{% endtabs %}

### 今後のリリース予定

リリース予定時期は変更になる可能性があります。

<table><thead><tr><th width="187">リリース予定時期</th><th width="262">リリース内容</th><th>利用制限</th></tr></thead><tbody><tr><td>未定</td><td>未定</td><td>未定</td></tr></tbody></table>
