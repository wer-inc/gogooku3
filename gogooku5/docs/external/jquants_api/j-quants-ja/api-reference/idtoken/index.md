# IDトークン取得(/token/auth\_refresh)

### API概要

リフレッシュトークンを用いてIDトークンを取得することができます。\
リフレッシュトークンはJ-Quants APIのユーザ登録およびサブスクリプションプランへの登録後、[メニュー画面](https://jpx-jquants.com/dashboard/menu)又は[リフレッシュトークン取得API](https://jpx.gitbook.io/j-quants-ja/api-reference/refreshtoken)から取得することができます。

### 本APIの留意点

{% hint style="info" %}
リフレッシュトークンの有効期間は１週間、IDトークンの有効期間は２４時間です。
{% endhint %}

### パラメータ及びレスポンス

## IDトークンを取得します

<mark style="color:green;">`POST`</mark> `https://api.jquants.com/v1/token/auth_refresh`

データの取得では、リフレッシュトークンの指定が必須となります。

\*は必須項目

#### Query Parameters

| Name                                           | Type   | Description |
| ---------------------------------------------- | ------ | ----------- |
| refreshtoken<mark style="color:red;">\*</mark> | String | リフレッシュトークン  |

#### Headers

| Name | Type | Description |
| ---- | ---- | ----------- |
| 不要   | 不要   | 不要          |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "idToken": "<YOUR idToken>"
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```json
{
    "message": "'refreshtoken' is required."
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
{% endtabs %}

### データ項目概要

| 変数名     | 説明     | 型      | 備考 |
| ------- | ------ | ------ | -- |
| idToken | IDトークン | String |    |

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
REFRESH_TOKEN=<YOUR REFRESH_TOKEN> && curl -XPOST https://api.jquants.com/v1/token/auth_refresh?refreshtoken=$REFRESH_TOKEN
```

{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" %}

```python
import requests
import json

REFRESH_TOKEN = "YOUR refreshtokenID"
r_post = requests.post(f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={REFRESH_TOKEN}")
r_post.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
