# 指数四本値(/indices)

### APIの概要

各種指数の四本値データを取得することができます。

現在配信している指数につきましては、[こちらのページ](https://jpx.gitbook.io/j-quants-ja/api-reference/indices/indexcodes)を参照ください。

### 本APIの留意点

{% hint style="info" %}

* 2022年4月の東証市場区分再編によりマザーズ市場は廃止されていますが、一定のルールに基づき東証マザーズ指数の構成銘柄の入替を行い、 2023年11月6日より指数名称を「東証グロース市場250指数」に変更されています。\
  詳細は[こちら](https://www.jpx.co.jp/news/6030/20230428-01.html)をご参照ください。
* 2020年10月1日のデータは東京証券取引所の株式売買システムの障害により終日売買停止となった関係で、四本値は前営業日（2020年10月1日）の終値が収録されています。
  {% endhint %}

### パラメータ及びレスポンス

データの取得する際には、指数コード（code）または日付（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="93" data-type="checkbox">code</th><th width="87" data-type="checkbox">date</th><th width="110" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>配信している指数全てについて指定された日付のデータ</td></tr></tbody></table>

## 日次の指数四本値データを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/indices`

データの取得では、指数コード（code）または日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                  |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| code            | String | <p>指数コード</p><p>（e.g. 0000 or 0028）</p><p>配信対象の指数コードについては<a href="indices/indexcodes">こちらのページ</a>を参照ください。</p> |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                                                           |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                             |
| date            | String | <p>\*fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                 |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                        |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "indices": [
        {
            "Date": "2023-12-01",
            "Code": "0028",
            "Open": 1199.18,
            "High": 1202.58,
            "Low": 1195.01,
            "Close": 1200.17
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

<table><thead><tr><th>変数名</th><th>説明</th><th width="109">型</th><th width="213">備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Code</td><td>指数コード</td><td>String</td><td>配信対象の指数コードは<a href="indices/indexcodes">こちらのページ</a>を参照ください。</td></tr><tr><td>Open</td><td>始値</td><td>Number</td><td></td></tr><tr><td>High</td><td>高値</td><td>Number</td><td></td></tr><tr><td>Low</td><td>安値</td><td>Number</td><td></td></tr><tr><td>Close</td><td>終値</td><td>Number</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/indices?code=0028&date=20231201 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/indices?code=0028&date=20231201", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
