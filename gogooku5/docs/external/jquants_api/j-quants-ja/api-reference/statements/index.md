# 財務情報(/fins/statements)

### APIの概要

上場企業の四半期毎の決算短信サマリーや業績・配当情報の修正に関する開示情報（主に数値データ）を取得することができます。

### 本APIの留意点

{% hint style="info" %}
**会計基準について**

APIから出力される各項目名は日本基準（JGAAP）の開示項目が基準となっています。

そのため、IFRSや米国基準（USGAAP）の開示データにおいては、経常利益の概念がありませんので、データが空欄となっています。
{% endhint %}

{% hint style="info" %}
**四半期開示見直し対応に伴うAPI項目の追加について**

四半期開示見直し対応において、決算短信サマリー様式の記載事項が以下のとおり変更されます。

* **変更前：**&#x91CD;要な⼦会社の異動（連結範囲の変更を伴う特定⼦会社の異動）
* **変更後：**&#x9023;結範囲の重要な変更

この対応に伴い、2024/7/22より本APIのレスポンス項目に"SignificantChangesInTheScopeOfConsolidation"（期中における連結範囲の重要な変更）を追加いたします。

詳細は、データ項目概要欄をご覧ください。
{% endhint %}

### パラメータ及びレスポンス

## 四半期の財務情報を取得することができます

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/fins/statements`

リクエストパラメータにcode（銘柄コード）またはdate（開示日）を入力する必要があります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| code            | String | <p>4桁もしくは5桁の銘柄コード</p><p>ex.86970 or 8697</p>                          |
| date            | String | ex.2022-01-05 or 20220105                                             |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "statements": [
        {
              "DisclosedDate": "2023-01-30",
              "DisclosedTime": "12:00:00",
              "LocalCode": "86970",
              "DisclosureNumber": "20230127594871",
              "TypeOfDocument": "3QFinancialStatements_Consolidated_IFRS",
              "TypeOfCurrentPeriod": "3Q",
              "CurrentPeriodStartDate": "2022-04-01",
              "CurrentPeriodEndDate": "2022-12-31",
              "CurrentFiscalYearStartDate": "2022-04-01",
              "CurrentFiscalYearEndDate": "2023-03-31",
              "NextFiscalYearStartDate": "",
              "NextFiscalYearEndDate": "",
              "NetSales": "100529000000",
              "OperatingProfit": "51765000000",
              "OrdinaryProfit": "",
              "Profit": "35175000000",
              "EarningsPerShare": "66.76",
              "DilutedEarningsPerShare": "",
              "TotalAssets": "79205861000000",
              "Equity": "320021000000",
              "EquityToAssetRatio": "0.004",
              "BookValuePerShare": "",
              "CashFlowsFromOperatingActivities": "",
              "CashFlowsFromInvestingActivities": "",
              "CashFlowsFromFinancingActivities": "",
              "CashAndEquivalents": "91135000000",
              "ResultDividendPerShare1stQuarter": "",
              "ResultDividendPerShare2ndQuarter": "26.0",
              "ResultDividendPerShare3rdQuarter": "",
              "ResultDividendPerShareFiscalYearEnd": "",
              "ResultDividendPerShareAnnual": "",
              "DistributionsPerUnit(REIT)": "",
              "ResultTotalDividendPaidAnnual": "",
              "ResultPayoutRatioAnnual": "",
              "ForecastDividendPerShare1stQuarter": "",
              "ForecastDividendPerShare2ndQuarter": "",
              "ForecastDividendPerShare3rdQuarter": "",
              "ForecastDividendPerShareFiscalYearEnd": "36.0",
              "ForecastDividendPerShareAnnual": "62.0",
              "ForecastDistributionsPerUnit(REIT)": "",
              "ForecastTotalDividendPaidAnnual": "",
              "ForecastPayoutRatioAnnual": "",
              "NextYearForecastDividendPerShare1stQuarter": "",
              "NextYearForecastDividendPerShare2ndQuarter": "",
              "NextYearForecastDividendPerShare3rdQuarter": "",
              "NextYearForecastDividendPerShareFiscalYearEnd": "",
              "NextYearForecastDividendPerShareAnnual": "",
              "NextYearForecastDistributionsPerUnit(REIT)": "",
              "NextYearForecastPayoutRatioAnnual": "",
              "ForecastNetSales2ndQuarter": "",
              "ForecastOperatingProfit2ndQuarter": "",
              "ForecastOrdinaryProfit2ndQuarter": "",
              "ForecastProfit2ndQuarter": "",
              "ForecastEarningsPerShare2ndQuarter": "",
              "NextYearForecastNetSales2ndQuarter": "",
              "NextYearForecastOperatingProfit2ndQuarter": "",
              "NextYearForecastOrdinaryProfit2ndQuarter": "",
              "NextYearForecastProfit2ndQuarter": "",
              "NextYearForecastEarningsPerShare2ndQuarter": "",
              "ForecastNetSales": "132500000000",
              "ForecastOperatingProfit": "65500000000",
              "ForecastOrdinaryProfit": "",
              "ForecastProfit": "45000000000",
              "ForecastEarningsPerShare": "85.42",
              "NextYearForecastNetSales": "",
              "NextYearForecastOperatingProfit": "",
              "NextYearForecastOrdinaryProfit": "",
              "NextYearForecastProfit": "",
              "NextYearForecastEarningsPerShare": "",
              "MaterialChangesInSubsidiaries": "false",
              "SignificantChangesInTheScopeOfConsolidation": "",
              "ChangesBasedOnRevisionsOfAccountingStandard": "false",
              "ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard": "false",
              "ChangesInAccountingEstimates": "true",
              "RetrospectiveRestatement": "",
              "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": "528578441",
              "NumberOfTreasuryStockAtTheEndOfFiscalYear": "1861043",
              "AverageNumberOfShares": "526874759",
              "NonConsolidatedNetSales": "",
              "NonConsolidatedOperatingProfit": "",
              "NonConsolidatedOrdinaryProfit": "",
              "NonConsolidatedProfit": "",
              "NonConsolidatedEarningsPerShare": "",
              "NonConsolidatedTotalAssets": "",
              "NonConsolidatedEquity": "",
              "NonConsolidatedEquityToAssetRatio": "",
              "NonConsolidatedBookValuePerShare": "",
              "ForecastNonConsolidatedNetSales2ndQuarter": "",
              "ForecastNonConsolidatedOperatingProfit2ndQuarter": "",
              "ForecastNonConsolidatedOrdinaryProfit2ndQuarter": "",
              "ForecastNonConsolidatedProfit2ndQuarter": "",
              "ForecastNonConsolidatedEarningsPerShare2ndQuarter": "",
              "NextYearForecastNonConsolidatedNetSales2ndQuarter": "",
              "NextYearForecastNonConsolidatedOperatingProfit2ndQuarter": "",
              "NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter": "",
              "NextYearForecastNonConsolidatedProfit2ndQuarter": "",
              "NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter": "",
              "ForecastNonConsolidatedNetSales": "",
              "ForecastNonConsolidatedOperatingProfit": "",
              "ForecastNonConsolidatedOrdinaryProfit": "",
              "ForecastNonConsolidatedProfit": "",
              "ForecastNonConsolidatedEarningsPerShare": "",
              "NextYearForecastNonConsolidatedNetSales": "",
              "NextYearForecastNonConsolidatedOperatingProfit": "",
              "NextYearForecastNonConsolidatedOrdinaryProfit": "",
              "NextYearForecastNonConsolidatedProfit": "",
              "NextYearForecastNonConsolidatedEarningsPerShare": ""
        }
    ],
    "pagination_key": "value1.value2."
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```json
{
　　　　　　　　"message": "This API requires at least 1 parameter as follows; 'date','code'."
}
```

{% endtab %}

{% tab title="401: Unauthorized " %}

```json
{
　　　　　　　　"message": "The incoming token is invalid or expired."
}
```

{% endtab %}

{% tab title="403: Forbidden " %}

```json
{
　　　　　　　　"message": <Error Message>
}
```

{% endtab %}

{% tab title="500: Internal Server Error " %}

```json
{
　　　　　　　　"message": "Unexpected error. Please try again later."
}
```

{% endtab %}

{% tab title="413: Payload Too Large " %}

```json
{
    "message": "Response data is too large. Specify parameters to reduce the acquired data range."
}
```

{% endtab %}
{% endtabs %}

### データ項目概要

<table><thead><tr><th>変数名</th><th>説明</th><th width="105">型</th><th>備考</th></tr></thead><tbody><tr><td>DisclosedDate</td><td>開示日</td><td>String</td><td></td></tr><tr><td>DisclosedTime</td><td>開示時刻</td><td>String</td><td></td></tr><tr><td>LocalCode</td><td>銘柄コード（5桁）</td><td>String</td><td></td></tr><tr><td>DisclosureNumber</td><td>開示番号</td><td>String</td><td>APIから出力されるjsonは開示番号で昇順に並んでいます。</td></tr><tr><td>TypeOfDocument</td><td>開示書類種別</td><td>String</td><td><a href="statements/typeofdocument">開示書類種別一覧</a></td></tr><tr><td>TypeOfCurrentPeriod</td><td>当会計期間の種類</td><td>String</td><td>[1Q, 2Q, 3Q, 4Q, 5Q, FY]</td></tr><tr><td>CurrentPeriodStartDate</td><td>当会計期間開始日</td><td>String</td><td></td></tr><tr><td>CurrentPeriodEndDate</td><td>当会計期間終了日</td><td>String</td><td></td></tr><tr><td>CurrentFiscalYearStartDate</td><td>当事業年度開始日</td><td>String</td><td></td></tr><tr><td>CurrentFiscalYearEndDate</td><td>当事業年度終了日</td><td>String</td><td></td></tr><tr><td>NextFiscalYearStartDate</td><td>翌事業年度開始日</td><td>String</td><td>開示レコードに翌事業年度の開示情報がない場合空欄になります。</td></tr><tr><td>NextFiscalYearEndDate</td><td>翌事業年度終了日</td><td>String</td><td>同上</td></tr><tr><td>NetSales</td><td>売上高</td><td>String</td><td></td></tr><tr><td>OperatingProfit</td><td>営業利益</td><td>String</td><td></td></tr><tr><td>OrdinaryProfit</td><td>経常利益</td><td>String</td><td></td></tr><tr><td>Profit</td><td>当期純利益</td><td>String</td><td></td></tr><tr><td>EarningsPerShare</td><td>一株あたり当期純利益</td><td>String</td><td></td></tr><tr><td>DilutedEarningsPerShare</td><td>潜在株式調整後一株あたり当期純利益</td><td>String</td><td></td></tr><tr><td>TotalAssets</td><td>総資産</td><td>String</td><td></td></tr><tr><td>Equity</td><td>純資産</td><td>String</td><td></td></tr><tr><td>EquityToAssetRatio</td><td>自己資本比率</td><td>String</td><td></td></tr><tr><td>BookValuePerShare</td><td>一株あたり純資産</td><td>String</td><td></td></tr><tr><td>CashFlowsFromOperatingActivities</td><td>営業活動によるキャッシュ・フロー</td><td>String</td><td></td></tr><tr><td>CashFlowsFromInvestingActivities</td><td>投資活動によるキャッシュ・フロー</td><td>String</td><td></td></tr><tr><td>CashFlowsFromFinancingActivities</td><td>財務活動によるキャッシュ・フロー</td><td>String</td><td></td></tr><tr><td>CashAndEquivalents</td><td>現金及び現金同等物期末残高</td><td>String</td><td></td></tr><tr><td>ResultDividendPerShare1stQuarter</td><td>一株あたり配当実績_第1四半期末</td><td>String</td><td></td></tr><tr><td>ResultDividendPerShare2ndQuarter</td><td>一株あたり配当実績_第2四半期末</td><td>String</td><td></td></tr><tr><td>ResultDividendPerShare3rdQuarter</td><td>一株あたり配当実績_第3四半期末</td><td>String</td><td></td></tr><tr><td>ResultDividendPerShareFiscalYearEnd</td><td>一株あたり配当実績_期末</td><td>String</td><td></td></tr><tr><td>ResultDividendPerShareAnnual</td><td>一株あたり配当実績_合計</td><td>String</td><td></td></tr><tr><td>DistributionsPerUnit(REIT)</td><td>1口当たり分配金</td><td>String</td><td></td></tr><tr><td>ResultTotalDividendPaidAnnual</td><td>配当金総額</td><td>String</td><td></td></tr><tr><td>ResultPayoutRatioAnnual</td><td>配当性向</td><td>String</td><td></td></tr><tr><td>ForecastDividendPerShare1stQuarter</td><td>一株あたり配当予想_第1四半期末</td><td>String</td><td></td></tr><tr><td>ForecastDividendPerShare2ndQuarter</td><td>一株あたり配当予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastDividendPerShare3rdQuarter</td><td>一株あたり配当予想_第3四半期末</td><td>String</td><td></td></tr><tr><td>ForecastDividendPerShareFiscalYearEnd</td><td>一株あたり配当予想_期末</td><td>String</td><td></td></tr><tr><td>ForecastDividendPerShareAnnual</td><td>一株あたり配当予想_合計</td><td>String</td><td></td></tr><tr><td>ForecastDistributionsPerUnit(REIT)</td><td>1口当たり予想分配金</td><td>String</td><td></td></tr><tr><td>ForecastTotalDividendPaidAnnual</td><td>予想配当金総額</td><td>String</td><td></td></tr><tr><td>ForecastPayoutRatioAnnual</td><td>予想配当性向</td><td>String</td><td></td></tr><tr><td>NextYearForecastDividendPerShare1stQuarter</td><td>一株あたり配当予想_翌事業年度第1四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastDividendPerShare2ndQuarter</td><td>一株あたり配当予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastDividendPerShare3rdQuarter</td><td>一株あたり配当予想_翌事業年度第3四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastDividendPerShareFiscalYearEnd</td><td>一株あたり配当予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastDividendPerShareAnnual</td><td>一株あたり配当予想_翌事業年度合計</td><td>String</td><td></td></tr><tr><td>NextYearForecastDistributionsPerUnit(REIT)</td><td>1口当たり翌事業年度予想分配金</td><td>String</td><td></td></tr><tr><td>NextYearForecastPayoutRatioAnnual</td><td>翌事業年度予想配当性向</td><td>String</td><td></td></tr><tr><td>ForecastNetSales2ndQuarter</td><td>売上高_予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastOperatingProfit2ndQuarter</td><td>営業利益_予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastOrdinaryProfit2ndQuarter</td><td>経常利益_予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastProfit2ndQuarter</td><td>当期純利益_予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastEarningsPerShare2ndQuarter</td><td>一株あたり当期純利益_予想_第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastNetSales2ndQuarter</td><td>売上高_予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastOperatingProfit2ndQuarter</td><td>営業利益_予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastOrdinaryProfit2ndQuarter</td><td>経常利益_予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastProfit2ndQuarter</td><td>当期純利益_予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastEarningsPerShare2ndQuarter</td><td>一株あたり当期純利益_予想_翌事業年度第2四半期末</td><td>String</td><td></td></tr><tr><td>ForecastNetSales</td><td>売上高_予想_期末</td><td>String</td><td></td></tr><tr><td>ForecastOperatingProfit</td><td>営業利益_予想_期末</td><td>String</td><td></td></tr><tr><td>ForecastOrdinaryProfit</td><td>経常利益_予想_期末</td><td>String</td><td></td></tr><tr><td>ForecastProfit</td><td>当期純利益_予想_期末</td><td>String</td><td></td></tr><tr><td>ForecastEarningsPerShare</td><td>一株あたり当期純利益_予想_期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastNetSales</td><td>売上高_予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastOperatingProfit</td><td>営業利益_予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastOrdinaryProfit</td><td>経常利益_予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastProfit</td><td>当期純利益_予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>NextYearForecastEarningsPerShare</td><td>一株あたり当期純利益_予想_翌事業年度期末</td><td>String</td><td></td></tr><tr><td>MaterialChangesInSubsidiaries</td><td>期中における重要な子会社の異動</td><td>String</td><td></td></tr><tr><td>SignificantChangesInTheScopeOfConsolidation</td><td>期中における連結範囲の重要な変更</td><td>String</td><td>*指定されたdateが2024-07-21以前のレスポンスは、当該項目には値が収録されません。</td></tr><tr><td>ChangesBasedOnRevisionsOfAccountingStandard</td><td>会計基準等の改正に伴う会計方針の変更</td><td>String</td><td></td></tr><tr><td>ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard</td><td>会計基準等の改正に伴う変更以外の会計方針の変更</td><td>String</td><td></td></tr><tr><td>ChangesInAccountingEstimates</td><td>会計上の見積りの変更</td><td>String</td><td></td></tr><tr><td>RetrospectiveRestatement</td><td>修正再表示</td><td>String</td><td></td></tr><tr><td>NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock</td><td>期末発行済株式数</td><td>String</td><td></td></tr><tr><td>NumberOfTreasuryStockAtTheEndOfFiscalYear</td><td>期末自己株式数</td><td>String</td><td></td></tr><tr><td>AverageNumberOfShares</td><td>期中平均株式数</td><td>String</td><td></td></tr><tr><td>NonConsolidatedNetSales</td><td>売上高_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedOperatingProfit</td><td>営業利益_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedOrdinaryProfit</td><td>経常利益_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedProfit</td><td>当期純利益_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedEarningsPerShare</td><td>一株あたり当期純利益_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedTotalAssets</td><td>総資産_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedEquity</td><td>純資産_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedEquityToAssetRatio</td><td>自己資本比率_非連結</td><td>String</td><td></td></tr><tr><td>NonConsolidatedBookValuePerShare</td><td>一株あたり純資産_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedNetSales2ndQuarter</td><td>売上高_予想_第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedOperatingProfit2ndQuarter</td><td>営業利益_予想_第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedOrdinaryProfit2ndQuarter</td><td>経常利益_予想_第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedProfit2ndQuarter</td><td>当期純利益_予想_第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedEarningsPerShare2ndQuarter</td><td>一株あたり当期純利益_予想_第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedNetSales2ndQuarter</td><td>売上高_予想_翌事業年度第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedOperatingProfit2ndQuarter</td><td>営業利益_予想_翌事業年度第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter</td><td>経常利益_予想_翌事業年度第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedProfit2ndQuarter</td><td>当期純利益_予想_翌事業年度第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter</td><td>一株あたり当期純利益_予想_翌事業年度第2四半期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedNetSales</td><td>売上高_予想_期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedOperatingProfit</td><td>営業利益_予想_期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedOrdinaryProfit</td><td>経常利益_予想_期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedProfit</td><td>当期純利益_予想_期末_非連結</td><td>String</td><td></td></tr><tr><td>ForecastNonConsolidatedEarningsPerShare</td><td>一株あたり当期純利益_予想_期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedNetSales</td><td>売上高_予想_翌事業年度期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedOperatingProfit</td><td>営業利益_予想_翌事業年度期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedOrdinaryProfit</td><td>経常利益_予想_翌事業年度期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedProfit</td><td>当期純利益_予想_翌事業年度期末_非連結</td><td>String</td><td></td></tr><tr><td>NextYearForecastNonConsolidatedEarningsPerShare</td><td>一株あたり当期純利益_予想_翌事業年度期末_非連結</td><td>String</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/fins/statements?code=86970&date=20230130 -H "Authorization: Bearer $idToken"
```

{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" %}

```python
import requests
import json

idToken = "YOUR idToken"
headers = {'Authorization': 'Bearer {}'.format(idToken)}
r = requests.get("https://api.jquants.com/v1/fins/statements?code=86970&date=20230130", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
