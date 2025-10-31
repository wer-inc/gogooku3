# リフレッシュトークン取得(/token/auth\_user)

### API概要

登録したメールアドレスとパスワードを用いてリフレッシュトークンを取得することができます。\
J-Quants APIのサービスに登録し、ログイン出来るメールアドレスとパスワードをご用意ください。

### 本APIの留意点

{% hint style="info" %}
本APIにより取得するリフレッシュトークンの有効期間は１週間です。
{% endhint %}

### パラメータ及びレスポンス

## リフレッシュトークンを取得します

<mark style="color:green;">`POST`</mark> `https://api.jquants.com/v1/token/auth_user`

データの取得では、メールアドレスとパスワードの指定が必須となります。

\*は必須項目

#### Headers

| Name | Type | Description |
| ---- | ---- | ----------- |
| 不要   | 不要   | 不要          |

#### Request Body

| Name                                          | Type   | Description |
| --------------------------------------------- | ------ | ----------- |
| mailaddress<mark style="color:red;">\*</mark> | String | メールアドレス     |
| password<mark style="color:red;">\*</mark>    | String | パスワード       |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "refreshToken": "<YOUR refreshToken>"
}
```

{% endtab %}

{% tab title="400: Bad Request " %}

```json
{
    "message": "'mailaddress' or 'password' is incorrect."
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

| 変数名          | 説明         | 型      | 備考 |
| ------------ | ---------- | ------ | -- |
| refreshToken | リフレッシュトークン | String |    |

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong>BODY="{\"mailaddress\":\"&#x3C;YOUR EMAIL_ADDRESS>\", \"password\":\"&#x3C;YOUR PASSWORD>\"}" &#x26;&#x26; curl -X POST -H "Content-Type: application/json" -d "$BODY" https://api.jquants.com/v1/token/auth_user
</strong></code></pre>

{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" %}

```python
import requests
import json

data={"mailaddress":"<YOUR EMAIL_ADDRESS>", "password":"<YOUR PASSWORD>"}
r_post = requests.post("https://api.jquants.com/v1/token/auth_user", data=json.dumps(data))
r_post.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
