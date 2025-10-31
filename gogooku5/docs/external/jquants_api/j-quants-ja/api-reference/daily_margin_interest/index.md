# 日々公表信用取引残高(/markets/daily\_margin\_interest)

### API概要

各銘柄についての信用取引残高（株数）を取得できます。

配信データは下記のページで公表している内容と同一です。\
<https://www.jpx.co.jp/markets/statistics-equities/margin/index.html>

### 本APIの留意点

{% hint style="info" %}

* 当該銘柄のコーポレートアクションが発生した場合であっても、遡及して約定株数の調整は行われません。
* 東京証券取引所または日本証券金融が、日次の信用取引残高を公表する必要があると認めた銘柄のみが収録されます。
* 過誤訂正により過去の日々公表信用取引残高データが訂正された場合は、本APIでは以下のとおりデータを提供します。
  * 訂正前と訂正後のデータのいずれも提供します。訂正が生じた場合には、申込日を同一とするレコードが追加されます。公表日が新しいデータが訂正後、公表日が古いデータが訂正前のデータと識別することが可能です。
    {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）または公表日（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table data-full-width="false"><thead><tr><th width="104.66666666666669" data-type="checkbox">code</th><th width="83" data-type="checkbox">date</th><th width="107" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>全上場銘柄について指定された公表日のデータ</td></tr></tbody></table>

## 日々公表信用取引残高を取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/daily_margin_interest`

データの取得では、銘柄コード（code）または公表日（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                                                          |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                            |
| date            | String | <p>fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                  |
| pagination\_key | String | 検索の先頭を指定する文字列過去の検索で返却された`pagination_key`を設定                                                                 |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "daily_margin_interest": [
        {
            "PublishedDate": "2024-02-08",
            "Code": "13260",
            "ApplicationDate": "2024-02-07",
            "PublishReason": {
                "Restricted": "0",
                "DailyPublication": "0",
                "Monitoring": "0",
                "RestrictedByJSF": "0",
                "PrecautionByJSF": "1",
                "UnclearOrSecOnAlert": "0"
            },
            "ShortMarginOutstanding": 11.0,
            "DailyChangeShortMarginOutstanding": 0.0,
            "ShortMarginOutstandingListedShareRatio": "*",
            "LongMarginOutstanding": 676.0,
            "DailyChangeLongMarginOutstanding": -20.0,
            "LongMarginOutstandingListedShareRatio": "*",
            "ShortLongRatio": 1.6,
            "ShortNegotiableMarginOutstanding": 0.0,
            "DailyChangeShortNegotiableMarginOutstanding": 0.0,
            "ShortStandardizedMarginOutstanding": 11.0,
            "DailyChangeShortStandardizedMarginOutstanding": 0.0,
            "LongNegotiableMarginOutstanding": 192.0,
            "DailyChangeLongNegotiableMarginOutstanding": -20.0,
            "LongStandardizedMarginOutstanding": 484.0,
            "DailyChangeLongStandardizedMarginOutstanding": 0.0,
            "TSEMarginBorrowingAndLendingRegulationClassification": "001"
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
{% code overflow="wrap" %}

```json
{
    "message": "The incoming token is invalid or expired."
}
```

{% endcode %}
{% endtab %}

{% tab title="403: Forbidden " %}
{% code overflow="wrap" %}

```json
{
    "message": <Error Message>
}
```

{% endcode %}
{% endtab %}

{% tab title="500: Internal Server Error " %}
{% code overflow="wrap" %}

```json
{
    "message": "Unexpected error. Please try again later."
}
```

{% endcode %}
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

<table><thead><tr><th width="186">変数名</th><th width="296">説明</th><th width="101">型</th><th width="251">備考</th></tr></thead><tbody><tr><td>PublishedDate</td><td>公表日</td><td>String</td><td></td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>ApplicationDate</td><td>申込日</td><td>String</td><td>YYYY-MM-DD<br>信用取引残高基準となる時点を表します。</td></tr><tr><td>PublishReason</td><td><a href="daily_margin_interest/publish-reason">公表の理由</a></td><td>Map</td><td></td></tr><tr><td>ShortMarginOutstanding</td><td>売合計信用残高</td><td>Number</td><td></td></tr><tr><td>DailyChangeShortMarginOutstanding</td><td>前日比 売合計信用残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>ShortMarginOutstandingListedShareRatio</td><td>上場比 売合計信用残高（単位：％）<br>売合計信用残高 ÷ 上場株数 × 100</td><td>Number/String</td><td>上場株数に対する売合計信用残高の割合です。ETFの場合、*を出力します。</td></tr><tr><td>LongMarginOutstanding</td><td>買合計信用残高</td><td>Number</td><td></td></tr><tr><td>DailyChangeLongMarginOutstanding</td><td>前日比 買合計信用残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>LongMarginOutstandingListedShareRatio</td><td>上場比 買合計信用残高（単位：％）<br>買合計信用残高 ÷ 上場株数 × 100</td><td>Number/String</td><td>上場株数に対する買合計信用残高の割合です。ETFの場合、*を出力します。</td></tr><tr><td>ShortLongRatio</td><td>取組比率（単位：％）<br>売合計信用残高 ÷ 買合計信用残高 × 100</td><td>Number</td><td></td></tr><tr><td>ShortNegotiableMarginOutstanding</td><td>一般信用取引売残高</td><td>Number</td><td>売合計信用残高 のうち、一般信用によるものです</td></tr><tr><td>DailyChangeShortNegotiableMarginOutstanding</td><td>前日比 一般信用取引売残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>ShortStandardizedMarginOutstanding</td><td>制度信用取引売残高</td><td>Number</td><td>売合計信用残高 のうち、制度信用によるものです</td></tr><tr><td>DailyChangeShortStandardizedMarginOutstanding</td><td>前日比 制度信用取引売残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>LongNegotiableMarginOutstanding</td><td>一般信用取引買残高</td><td>Number</td><td>買合計信用残高 のうち、一般信用によるものです</td></tr><tr><td>DailyChangeLongNegotiableMarginOutstanding</td><td>前日比 一般信用取引買残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>LongStandardizedMarginOutstanding</td><td>制度信用取引買残高</td><td>Number</td><td>買合計信用残高 のうち、制度信用によるものです</td></tr><tr><td>DailyChangeLongStandardizedMarginOutstanding</td><td>前日比 制度信用取引買残高（単位：株）</td><td>Number/String</td><td>前日に公表されていない銘柄の場合、-を出力します。</td></tr><tr><td>TSEMarginBorrowingAndLendingRegulationClassification</td><td><a href="daily_margin_interest/margin_trading_classification">東証信用貸借規制区分</a></td><td>String</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/daily_margin_interest?code=86970&date=20230324 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/daily_margin_interest?code=86970&date=20230324", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
