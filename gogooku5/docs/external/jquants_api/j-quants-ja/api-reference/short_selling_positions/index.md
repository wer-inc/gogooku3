# 空売り残高報告(/markets/short\_selling\_positions)

### API概要

「有価証券の取引等の規制に関する内閣府令」に基づき、取引参加者より報告を受けたもののうち、残高割合が0.5％以上のものについての情報を取得できます。

配信データは下記のページで公表している内容と同一ですが、より長いヒストリカルデータを利用可能です。\
<https://www.jpx.co.jp/markets/public/short-selling/index.html>

### 本APIの留意点

{% hint style="info" %}

* 取引参加者から該当する報告が行われなかった日にはデータは提供されません。
* 「有価証券の取引等の規制に関する内閣府令について」はこちらをご覧ください。<https://www.jpx.co.jp/markets/public/short-selling/01.html>
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）、公表日（disclosed\_date）、計算日（calculated\_date）のいずれかの指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="83" data-type="checkbox">code</th><th width="126" data-type="checkbox">disclosed_date</th><th width="160" data-type="checkbox">disclosed_date_from/disclosed_date_to</th><th width="126" data-type="checkbox">calculated_date</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>true</td><td>false</td><td>false</td><td>指定された銘柄について指定日（公表日）のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>false</td><td>指定された銘柄について指定された期間のデータ</td></tr><tr><td>true</td><td>false</td><td>false</td><td>true</td><td>指定された銘柄について指定日（計算日）のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>false</td><td>指定日（公表日）の全ての銘柄のデータ</td></tr><tr><td>false</td><td>false</td><td>false</td><td>true</td><td>指定日（計算日）の全ての銘柄のデータ</td></tr></tbody></table>

## 空売り残高報告データを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/short_selling_positions`

データの取得では、銘柄コード（code）、公表日（disclosed\_date）、計算日（caldulated\_date）のいずれかの指定が必須となります。

\*は必須項目

#### Query Parameters

| Name                  | Type   | Description                                                                                                          |
| --------------------- | ------ | -------------------------------------------------------------------------------------------------------------------- |
| code                  | String | <p>4桁もしくは5桁の銘柄コード</p><p>（e.g. 8697 or 86970）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| disclosed\_date       | String | <p>公表日の指定</p><p>（e.g. 20240301 or 2024-03-01）</p>                                                                    |
| disclosed\_date\_from | String | <p>公表日のfromの指定</p><p>（e.g. 20240301 or 2024-03-01）</p>                                                               |
| disclosed\_date\_to   | String | <p>公表日のtoの指定</p><p>（e.g. 20240301 or 2024-03-01）</p>                                                                 |
| calculated\_date      | String | <p>計算日の指定</p><p>（e.g. 20240301 or 2024-03-01）</p>                                                                    |
| pagination\_key       | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                                |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "short_selling_positions": [
      {
        "DisclosedDate": "2024-08-01",
        "CalculatedDate": "2024-07-31",
        "Code": "13660",
        "ShortSellerName": "個人",
        "ShortSellerAddress": "",
        "DiscretionaryInvestmentContractorName": "",
        "DiscretionaryInvestmentContractorAddress": "",
        "InvestmentFundName": "",
        "ShortPositionsToSharesOutstandingRatio": 0.0053,
        "ShortPositionsInSharesNumber": 140000,
        "ShortPositionsInTradingUnitsNumber": 140000,
        "CalculationInPreviousReportingDate": "2024-07-22",
        "ShortPositionsInPreviousReportingRatio": 0.0043,
        "Notes": ""
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

```json
{
    "message": "Response data is too large. Specify parameters to reduce the acquired data range."
}
```

{% endtab %}
{% endtabs %}

### データ項目概要

<table><thead><tr><th>変数名</th><th>説明</th><th width="105">型</th><th>備考</th></tr></thead><tbody><tr><td>DisclosedDate</td><td>日付（公表日）</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>CalculatedDate</td><td>日付（計算日）</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td>5桁コード</td></tr><tr><td>ShortSellerName</td><td>商号・名称・氏名</td><td>String</td><td>"ShortSellerName"は、取引参加者から報告されたものをそのまま記載しているため、日本語名称または英語名称が混在しています。</td></tr><tr><td>ShortSellerAddress</td><td>住所・所在地</td><td>String</td><td></td></tr><tr><td>DiscretionaryInvestmentContractorName</td><td>委託者・投資一任契約の相手方の商号・名称・氏名</td><td>String</td><td></td></tr><tr><td>DiscretionaryInvestmentContractorAddress</td><td>委託者・投資一任契約の相手方の住所・所在地</td><td>String</td><td></td></tr><tr><td>InvestmentFundName</td><td>信託財産・運用財産の名称</td><td>String</td><td></td></tr><tr><td>ShortPositionsToSharesOutstandingRatio</td><td>空売り残高割合</td><td>Number</td><td></td></tr><tr><td>ShortPositionsInSharesNumber</td><td>空売り残高数量</td><td>Number</td><td></td></tr><tr><td>ShortPositionsInTradingUnitsNumber</td><td>空売り残高売買単位数</td><td>Number</td><td></td></tr><tr><td>CalculationInPreviousReportingDate</td><td>直近計算年月日</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>ShortPositionsInPreviousReportingRatio</td><td>直近空売り残高割合</td><td>Number</td><td></td></tr><tr><td>Notes</td><td>備考</td><td>String</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/short_selling_positions?code=86970&calculated_date=20240801 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/short_selling_positions?code=86970&calculated_date=20240801", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
