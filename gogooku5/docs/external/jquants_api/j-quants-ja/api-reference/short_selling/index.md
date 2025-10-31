# 業種別空売り比率(/markets/short\_selling)

### API概要

日々の業種（セクター）別の空売り比率に関する売買代金を取得できます。\
配信データは下記のページで公表している内容と同様です。\
<https://www.jpx.co.jp/markets/statistics-equities/short-selling/index.html>\
Webページでの公表値は百万円単位に丸められておりますが、APIでは円単位のデータとなります。

### 本APIの留意点

{% hint style="info" %}

* 取引高が存在しない（売買されていない）日の日付（date）を指定した場合は、値は空です。
* 2020/10/1は東京証券取引所の株式売買システムの障害により終日売買停止となった関係で、データが存在しません。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、日付（date）または33業種コード（sector33code）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="178" data-type="checkbox">sector33code</th><th width="87" data-type="checkbox">date</th><th width="110" data-type="checkbox">from/to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>false</td><td>true</td><td>false</td><td>全業種コードについて指定された日付のデータ</td></tr><tr><td>true</td><td>false</td><td>false</td><td>指定された業種コードについて、全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された業種コードについて指定された期間分のデータ</td></tr><tr><td>true</td><td>true</td><td>false</td><td>指定された業種コードについて指定された日付のデータ</td></tr></tbody></table>

## 日々の業種（セクター）別の空売り比率・売買代金を取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/short_selling`

データの取得では、33業種コード（sector33code）または日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| sector33code    | String | <p>33業種コード</p><p>（e.g. 0050 or 50）</p>                                |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                    |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                      |
| date            | String | <p>\*fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>          |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "short_selling": [
        {
            "Date": "2022-10-25",
            "Sector33Code": "0050",
            "SellingExcludingShortSellingTurnoverValue": 1333126400.0,
            "ShortSellingWithRestrictionsTurnoverValue": 787355200.0,
            "ShortSellingWithoutRestrictionsTurnoverValue": 149084300.0
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

<table><thead><tr><th>変数名</th><th>説明</th><th width="109">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Sector33Code</td><td>33業種コード</td><td>String</td><td><a href="listed_info/sector33code">33業種コード及び業種名</a>を参照</td></tr><tr><td>SellingExcludingShortSellingTurnoverValue</td><td>実注文の売買代金</td><td>Number</td><td></td></tr><tr><td>ShortSellingWithRestrictionsTurnoverValue</td><td>価格規制有りの空売り売買代金</td><td>Number</td><td></td></tr><tr><td>ShortSellingWithoutRestrictionsTurnoverValue</td><td>価格規制無しの空売り売買代金</td><td>Number</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/short_selling?sector33code=0050&date=2022-10-25 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/short_selling?sector33code=0050&date=2022-10-25", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
