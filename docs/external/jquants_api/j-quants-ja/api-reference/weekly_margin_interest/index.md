# 信用取引週末残高(/markets/weekly\_margin\_interest)

### API概要

週末時点での、各銘柄についての信用取引残高（株数）を取得できます。

配信データは下記のページで公表している内容と同一です。\
<https://www.jpx.co.jp/markets/statistics-equities/margin/05.html>

### 本APIの留意点

{% hint style="info" %}

* 当該銘柄のコーポレートアクションが発生した場合も、遡及して株数の調整は行われません。
* 年末年始など、営業日が2日以下の週はデータが提供されません。
* 東証上場銘柄でない銘柄（地方取引所単独上場銘柄）についてはデータの収録対象外となっております。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="101" data-type="checkbox">code</th><th width="110" data-type="checkbox">date</th><th width="128" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>全上場銘柄について指定された日付のデータ</td></tr></tbody></table>

## 週次の信用取引残高を取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/weekly_margin_interest`

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                                                          |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                            |
| date            | String | <p>\*fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                       |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "weekly_margin_interest": [
        {
            "Date": "2023-02-17",
            "Code": "13010",
            "ShortMarginTradeVolume": 4100.0,
            "LongMarginTradeVolume": 27600.0,
            "ShortNegotiableMarginTradeVolume": 1300.0,
            "LongNegotiableMarginTradeVolume": 7600.0,
            "ShortStandardizedMarginTradeVolume": 2800.0,
            "LongStandardizedMarginTradeVolume": 20000.0,
            "IssueType": "2"
        }
    ],
    "pagination_key": "value1.value2."
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```javascript
{
    "message": <Error Message>
}
```

{% endtab %}

{% tab title="401: Unauthorized " %}

```javascript
{
    "message": "The incoming token is invalid or expired."
}
```

{% endtab %}

{% tab title="403: Forbidden " %}

```javascript
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

| 変数名                                | 説明          | 型      | 備考                                                |
| ---------------------------------- | ----------- | ------ | ------------------------------------------------- |
| Date                               | 申込日付        | String | <p>YYYY-MM-DD<br>信用取引残高基準となる時点を表します。（通常は金曜日付）</p> |
| Code                               | 銘柄コード       | String |                                                   |
| ShortMarginTradeVolume             | 売合計信用取引週末残高 | Number |                                                   |
| LongMarginTradeVolume              | 買合計信用取引週末残高 | Number |                                                   |
| ShortNegotiableMarginTradeVolume   | 売一般信用取引週末残高 | Number | 売合計信用取引週末残高 のうち、一般信用によるものです                       |
| LongNegotiableMarginTradeVolume    | 買一般信用取引週末残高 | Number | 買合計信用取引週末残高 のうち、一般信用によるものです                       |
| ShortStandardizedMarginTradeVolume | 売制度信用取引週末残高 | Number | 売合計信用取引週末残高 のうち、制度信用によるものです                       |
| LongStandardizedMarginTradeVolume  | 買制度信用取引週末残高 | Number | 買合計信用取引週末残高 のうち、制度信用によるものです                       |
| IssueType                          | 銘柄区分        | String | 1: 信用銘柄、2: 貸借銘柄、3: その他                            |

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/weekly_margin_interest?code=86970 -H "Authorization: Bearer $idToken" 
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
r = requests.get("https://api.jquants.com/v1/markets/weekly_margin_interest?code=86970", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
