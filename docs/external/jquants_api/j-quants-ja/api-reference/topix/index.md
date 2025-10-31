# TOPIX指数四本値(/indices/topix)

### API概要

TOPIXの日通しの四本値を取得できます。\
本APIで取得可能な指数データはTOPIX（東証株価指数）のみとなります。

### パラメータ及びレスポンス

## 日次のTOPIX指数データ

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/indices/topix`

日付の範囲（from/to）を指定することができます。なお、指定しない場合は全期間のデータがレスポンスに収録されます。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                    |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                      |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}
{% code overflow="wrap" %}

```json
{
    "topix": [
        {
            "Date": "2022-06-28",
            "Open": 1885.52,
            "High": 1907.38,
            "Low": 1885.32,
            "Close": 1907.38,
        }
    ],
    "pagination_key": "value1.value2."
}
```

{% endcode %}
{% endtab %}

{% tab title="400: Bad Request " %}
{% code overflow="wrap" %}

```json
{
    "message": "'from' must be older than 'to'"
}
```

{% endcode %}
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

```json
{
    "message": "Response data is too large. Specify parameters to reduce the acquired data range."
}
```

{% endtab %}
{% endtabs %}

### データ項目概要

<table><thead><tr><th width="130">変数名</th><th width="109">説明</th><th>型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Open</td><td>始値</td><td>Number</td><td></td></tr><tr><td>High</td><td>高値</td><td>Number</td><td></td></tr><tr><td>Low</td><td>安値</td><td>Number</td><td></td></tr><tr><td>Close</td><td>終値</td><td>Number</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/indices/topix -H "Authorization: Bearer $idToken" 
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
r = requests.get("https://api.jquants.com/v1/indices/topix", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
