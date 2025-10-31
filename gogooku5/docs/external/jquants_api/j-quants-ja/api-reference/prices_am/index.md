# 前場四本値(/prices/prices\_am)

### API概要

前場終了時に、前場の株価データを取得することができます。

### 本APIの留意点

{% hint style="info" %}

* 前場の取引高が存在しない（売買されていない）銘柄についての四本値、取引高と売買代金は、Nullが収録されています。
* 東証上場銘柄でない銘柄（地方取引所単独上場銘柄）についてはデータの収録対象外となっております。
* なお、当日のデータは翌日6:00頃まで取得可能です。\
  ヒストリカルの前場四本値については[株価四本値](https://jpx.gitbook.io/j-quants-ja/api-reference/daily_quotes)をご利用ください
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）の指定が可能です\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="93" data-type="checkbox">Code</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>指定された銘柄についての前場の株価データ</td></tr><tr><td>false</td><td>全上場銘柄について前場の株価データ</td></tr></tbody></table>

## 前場の株価データを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/prices/prices_am`

データの取得では、銘柄コード（code）が指定できます。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                       |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "prices_am": [
        {
            "Date": "2023-03-20",
            "Code": "39400",
            "MorningOpen": 232.0,
            "MorningHigh": 244.0,
            "MorningLow": 232.0,
            "MorningClose": 240.0,
            "MorningVolume": 52600.0,
            "MorningTurnoverValue": 12518800.0
        }
    ],
    "pagination_key": "value1.value2."
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```json
{
　　　　"message": "'code' must be 4 or 5 digits."
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

{% tab title="210 取得可能時間外または存在しない銘柄コード" %}

```json
{
　　　　"message": "No content due to outside the target time or a stock code does not exist."
}
```

{% endtab %}
{% endtabs %}

### データ項目概要

<table><thead><tr><th>変数名</th><th width="174">説明</th><th width="121">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YY-MM-DD</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>MorningOpen</td><td>前場始値</td><td>Number</td><td></td></tr><tr><td>MorningHigh</td><td>前場高値</td><td>Number</td><td></td></tr><tr><td>MorningLow</td><td>前場安値</td><td>Number</td><td></td></tr><tr><td>MorningClose</td><td>前場終値</td><td>Number</td><td></td></tr><tr><td>MorningVolume</td><td>前場売買高</td><td>Number</td><td></td></tr><tr><td>MorningTurnoverValue</td><td>前場取引代金</td><td>Number</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/prices/prices_am?code=39400 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/prices/prices_am?code=39400", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
