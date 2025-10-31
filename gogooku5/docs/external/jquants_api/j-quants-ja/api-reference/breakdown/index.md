# 売買内訳データ(/markets/breakdown)

### API概要

東証上場銘柄の東証市場における銘柄別の日次売買代金・売買高（立会内取引 に限る）について、信用取引や空売りの利用に関する発注時のフラグ情報を用いて細分化したデータです。

### 本APIの留意点

{% hint style="info" %}

* 当該銘柄のコーポレートアクションが発生した場合も、遡及して約定株数の調整は行われません。
* 2020/10/1は東京証券取引所の株式売買システムの障害により終日売買停止となった関係で、データが存在しません。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="93" data-type="checkbox">code</th><th width="87" data-type="checkbox">date</th><th width="110" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>全上場銘柄について指定された日付のデータ</td></tr></tbody></table>

## 銘柄別の日次売買代金・売買高のデータを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/breakdown`

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                                                          |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                            |
| date            | String | <p>fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                  |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                       |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "breakdown": [
        {
            "Date": "2015-04-01",
            "Code": "13010",
            "LongSellValue": 115164000.0,
            "ShortSellWithoutMarginValue": 93561000.0,
            "MarginSellNewValue": 6412000.0,
            "MarginSellCloseValue": 23009000.0,
            "LongBuyValue": 185114000.0,
            "MarginBuyNewValue": 35568000.0,
            "MarginBuyCloseValue": 17464000.0,
            "LongSellVolume": 415000.0,
            "ShortSellWithoutMarginVolume": 337000.0,
            "MarginSellNewVolume": 23000.0,
            "MarginSellCloseVolume": 83000.0,
            "LongBuyVolume": 667000.0,
            "MarginBuyNewVolume": 128000.0,
            "MarginBuyCloseVolume": 63000.0
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

<table><thead><tr><th width="172">変数名</th><th width="243">説明</th><th width="102">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>売買日</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>LongSellValue</td><td>実売りの約定代金</td><td>Number</td><td>売りの約定代金の内訳</td></tr><tr><td>ShortSellWithoutMarginValue</td><td>空売り(信用新規売りを除く)の約定代金</td><td>Number</td><td>同上</td></tr><tr><td>MarginSellNewValue</td><td>信用新規売り(新たな信用売りポジションを作るための売り注文)の約定代金</td><td>Number</td><td>同上</td></tr><tr><td>MarginSellCloseValue</td><td>信用返済売り(既存の信用買いポジションを閉じるための売り注文)の約定代金</td><td>Number</td><td>同上</td></tr><tr><td>LongBuyValue</td><td>現物買いの約定代金</td><td>Number</td><td>買いの約定代金の内訳</td></tr><tr><td>MarginBuyNewValue</td><td>信用新規買い(新たな信用買いポジションを作るための買い注文)の約定代金</td><td>Number</td><td>同上</td></tr><tr><td>MarginBuyCloseValue</td><td>信用返済買い(既存の信用売りポジションを閉じるための買い注文)の約定代金</td><td>Number</td><td>同上</td></tr><tr><td>LongSellVolume</td><td>実売りの約定株数</td><td>Number</td><td>売りの約定株数の内訳</td></tr><tr><td>ShortSellWithoutMarginVolume</td><td>空売り(信用新規売りを除く)の約定株数</td><td>Number</td><td>同上</td></tr><tr><td>MarginSellNewVolume</td><td>信用新規売り(新たな信用売りポジションを作るための売り注文)の約定株数</td><td>Number</td><td>同上</td></tr><tr><td>MarginSellCloseVolume</td><td>信用返済売り(既存の信用買いポジションを閉じるための売り注文)の約定株数</td><td>Number</td><td>同上</td></tr><tr><td>LongBuyVolume</td><td>現物買いの約定株数</td><td>Number</td><td>買いの約定株数の内訳</td></tr><tr><td>MarginBuyNewVolume</td><td>信用新規買い(新たな信用買いポジションを作るための買い注文)の約定株数</td><td>Number</td><td>同上</td></tr><tr><td>MarginBuyCloseVolume</td><td>信用返済買い(既存の信用売りポジションを閉じるための買い注文)の約定株数</td><td>Number</td><td>同上　</td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/breakdown?code=86970&date=20230324 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/breakdown?code=86970&date=20230324", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
