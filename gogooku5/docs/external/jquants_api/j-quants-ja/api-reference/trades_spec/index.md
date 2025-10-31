# 投資部門別情報(/markets/trades\_spec)

## API概要

投資部門別売買状況（株式・金額）のデータを取得することができます。\
配信データは下記のページで公表している内容と同一です。データの単位は千円です。\
<https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html>

### 本APIの留意点

{% hint style="info" %}

* 2022年4月4日に行われた市場区分見直しに伴い、市場区分に応じた内容となっている統計資料は、 見直し後の市場区分に変更して掲載しています。
* 過誤訂正により過去の投資部門別売買状況データが訂正された場合は、本APIでは以下のとおりデータを提供します。
  * 2023年4月3日以前に訂正が公表された過誤訂正：訂正前のデータは提供せず、訂正後のデータのみ提供します。
  * 2023年4月3日以降に訂正が公表された過誤訂正：訂正前と訂正後のデータのいずれも提供します。訂正が生じた場合には、市場名、開始日および終了日を同一とするレコードが追加され、公表日が新しいデータが訂正後、公表日が古いデータが訂正前のデータと識別することが可能です。
* 過誤訂正により過去の投資部門別売買状況データが訂正された場合は、過誤訂正が公表された翌営業日にデータが更新されます。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、セクション（section）または日付（from/to）の指定が可能です。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="133.66666666666669" data-type="checkbox">section</th><th width="143" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>true</td><td>指定したセクションの指定した期間のデータ</td></tr><tr><td>true</td><td>false</td><td>指定したセクションの全期間のデータ</td></tr><tr><td>false</td><td>true</td><td>すべてのセクションの指定した期間のデータ　</td></tr><tr><td>false</td><td>false</td><td>すべてのセクションの全期間のデータ</td></tr></tbody></table>

セクション（section）で指定可能なパラメータについてはこちらを参照ください。

{% content-ref url="trades\_spec/section" %}
[section](https://jpx.gitbook.io/j-quants-ja/api-reference/trades_spec/section)
{% endcontent-ref %}

## 投資部門別売買状況のデータを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/trades_spec`

データの取得では、市場（section）または公表日の日付（from/to）が指定できます。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| section         | String | <p>セクション</p><p>（e.g. TSEPrime）</p>                                    |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                    |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                      |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "trades_spec": [
        {
            "PublishedDate":"2017-01-13",
            "StartDate":"2017-01-04",
            "EndDate":"2017-01-06",
            "Section":"TSE1st",
            "ProprietarySales":1311271004,
            "ProprietaryPurchases":1453326508,
            "ProprietaryTotal":2764597512,
            "ProprietaryBalance":142055504,
            "BrokerageSales":7165529005,
            "BrokeragePurchases":7030019854,
            "BrokerageTotal":14195548859,
            "BrokerageBalance":-135509151,
            "TotalSales":8476800009,
            "TotalPurchases":8483346362,
            "TotalTotal":16960146371,
            "TotalBalance":6546353,
            "IndividualsSales":1401711615,
            "IndividualsPurchases":1161801155,
            "IndividualsTotal":2563512770,
            "IndividualsBalance":-239910460,
            "ForeignersSales":5094891735,
            "ForeignersPurchases":5317151774,
            "ForeignersTotal":10412043509,
            "ForeignersBalance":222260039,
            "SecuritiesCosSales":76381455,
            "SecuritiesCosPurchases":61700100,
            "SecuritiesCosTotal":138081555,
            "SecuritiesCosBalance":-14681355,
            "InvestmentTrustsSales":168705109,
            "InvestmentTrustsPurchases":124389642,
            "InvestmentTrustsTotal":293094751,
            "InvestmentTrustsBalance":-44315467,
            "BusinessCosSales":71217959,
            "BusinessCosPurchases":63526641,
            "BusinessCosTotal":134744600,
            "BusinessCosBalance":-7691318,
            "OtherCosSales":10745152,
            "OtherCosPurchases":15687836,
            "OtherCosTotal":26432988,
            "OtherCosBalance":4942684,
            "InsuranceCosSales":15926202,
            "InsuranceCosPurchases":9831555,
            "InsuranceCosTotal":25757757,
            "InsuranceCosBalance":-6094647,
            "CityBKsRegionalBKsEtcSales":10606789,
            "CityBKsRegionalBKsEtcPurchases":8843871,
            "CityBKsRegionalBKsEtcTotal":19450660,
            "CityBKsRegionalBKsEtcBalance":-1762918,
            "TrustBanksSales":292932297,
            "TrustBanksPurchases":245322795,
            "TrustBanksTotal":538255092,
            "TrustBanksBalance":-47609502,
            "OtherFinancialInstitutionsSales":22410692,
            "OtherFinancialInstitutionsPurchases":21764485,
            "OtherFinancialInstitutionsTotal":44175177,
            "OtherFinancialInstitutionsBalance":-646207
        }
    ],
    "pagination_key": "value1.value2."
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```json
{
    "message": <Error Message>
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
{% code overflow="wrap" %}

```json
{
    "message": "Response data is too large. Specify parameters to reduce the acquired data range."
}
```

{% endcode %}
{% endtab %}
{% endtabs %}

### データ項目概要

| 変数名                                 | 説明          | 型      | 備考                                                                             |
| ----------------------------------- | ----------- | ------ | ------------------------------------------------------------------------------ |
| PublishedDate                       | 公表日         | String | YY-MM-DD                                                                       |
| StartDate                           | 開始日         | String | YY-MM-DD                                                                       |
| EndDate                             | 終了日         | String | YY-MM-DD                                                                       |
| Section                             | 市場名         | String | [市場名](https://jpx.gitbook.io/j-quants-ja/api-reference/trades_spec/section)を参照 |
| ProprietarySales                    | 自己計\_売      | Number |                                                                                |
| ProprietaryPurchases                | 自己計\_買      | Number |                                                                                |
| ProprietaryTotal                    | 自己計\_合計     | Number |                                                                                |
| ProprietaryBalance                  | 自己計\_差引     | Number |                                                                                |
| BrokerageSales                      | 委託計\_売      | Number |                                                                                |
| BrokeragePurchases                  | 委託計\_買      | Number |                                                                                |
| BrokerageTotal                      | 委託計\_合計     | Number |                                                                                |
| BrokerageBalance                    | 委託計\_差引     | Number |                                                                                |
| TotalSales                          | 総計\_売       | Number |                                                                                |
| TotalPurchases                      | 総計\_買       | Number |                                                                                |
| TotalTotal                          | 総計\_合計      | Number |                                                                                |
| TotalBalance                        | 総計\_差引      | Number |                                                                                |
| IndividualsSales                    | 個人\_売       | Number |                                                                                |
| IndividualsPurchases                | 個人\_買       | Number |                                                                                |
| IndividualsTotal                    | 個人\_合計      | Number |                                                                                |
| IndividualsBalance                  | 個人\_差引      | Number |                                                                                |
| ForeignersSales                     | 海外投資家\_売    | Number |                                                                                |
| ForeignersPurchases                 | 海外投資家\_買    | Number |                                                                                |
| ForeignersTotal                     | 海外投資家\_合計   | Number |                                                                                |
| ForeignersBalance                   | 海外投資家\_差引   | Number |                                                                                |
| SecuritiesCosSales                  | 証券会社\_売     | Number |                                                                                |
| SecuritiesCosPurchases              | 証券会社\_買     | Number |                                                                                |
| SecuritiesCosTotal                  | 証券会社\_合計    | Number |                                                                                |
| SecuritiesCosBalance                | 証券会社\_差引    | Number |                                                                                |
| InvestmentTrustsSales               | 投資信託\_売     | Number |                                                                                |
| InvestmentTrustsPurchases           | 投資信託\_買     | Number |                                                                                |
| InvestmentTrustsTotal               | 投資信託\_合計    | Number |                                                                                |
| InvestmentTrustsBalance             | 投資信託\_差引    | Number |                                                                                |
| BusinessCosSales                    | 事業法人\_売     | Number |                                                                                |
| BusinessCosPurchases                | 事業法人\_買     | Number |                                                                                |
| BusinessCosTotal                    | 事業法人\_合計    | Number |                                                                                |
| BusinessCosBalance                  | 事業法人\_差引    | Number |                                                                                |
| OtherCosSales                       | その他法人\_売    | Number |                                                                                |
| OtherCosPurchases                   | その他法人\_買　   | Number |                                                                                |
| OtherCosTotal                       | その他法人\_合計   | Number |                                                                                |
| OtherCosBalance                     | その他法人\_差引   | Number |                                                                                |
| InsuranceCosSales                   | 生保・損保\_売    | Number |                                                                                |
| InsuranceCosPurchases               | 生保・損保\_買    | Number |                                                                                |
| InsuranceCosTotal                   | 生保・損保\_合計   | Number |                                                                                |
| InsuranceCosBalance                 | 生保・損保\_差引   | Number |                                                                                |
| CityBKsRegionalBKsEtcSales          | 都銀・地銀等\_売   | Number |                                                                                |
| CityBKsRegionalBKsEtcPurchases      | 都銀・地銀等\_買   | Number |                                                                                |
| CityBKsRegionalBKsEtcTotal          | 都銀・地銀等\_合計  | Number |                                                                                |
| CityBKsRegionalBKsEtcBalance        | 都銀・地銀等\_差引  | Number |                                                                                |
| TrustBanksSales                     | 信託銀行\_売     | Number |                                                                                |
| TrustBanksPurchases                 | 信託銀行\_買     | Number |                                                                                |
| TrustBanksTotal                     | 信託銀行\_合計    | Number |                                                                                |
| TrustBanksBalance                   | 信託銀行\_差引    | Number |                                                                                |
| OtherFinancialInstitutionsSales     | その他金融機関\_売  | Number |                                                                                |
| OtherFinancialInstitutionsPurchases | その他金融機関\_買  | Number |                                                                                |
| OtherFinancialInstitutionsTotal     | その他金融機関\_合計 | Number |                                                                                |
| OtherFinancialInstitutionsBalance   | その他金融機関\_差引 | Number |                                                                                |

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/trades_spec?section=TSEPrime&from=20230324&to=20230403 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/trades_spec?section=TSEPrime&from=20230324&to=20230403", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
