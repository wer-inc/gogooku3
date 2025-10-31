# 株価四本値(/prices/daily\_quotes)

### APIの概要

株価データを取得することができます。

株価は分割・併合を考慮した調整済み株価（小数点第２位四捨五入）と調整前の株価を取得することができます。

### 本APIの留意点

{% hint style="info" %}

* 取引高が存在しない（売買されていない）日の銘柄についての四本値、取引高と売買代金は、Nullが収録されています。
* 東証上場銘柄でない銘柄（地方取引所単独上場銘柄）についてはデータの収録対象外となっております。
* 2020/10/1のデータは東京証券取引所の株式売買システムの障害により終日売買停止となった関係で、四本値、取引高と売買代金はNullが収録されています。
* 日通しデータについては全プランで取得できますが、前場/後場別のデータについてはPremiumプランのみ取得可能です。
* 株価調整については株式分割・併合にのみ対応しております。一部コーポレートアクションには対応しておりませんので、ご了承ください。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="93" data-type="checkbox">code</th><th width="87" data-type="checkbox">date</th><th width="110" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について全期間分のデータ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>全上場銘柄について指定された日付のデータ</td></tr></tbody></table>

## 日次の株価データを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/prices/daily_quotes`

データの取得では、銘柄コード（code）または日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                  |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式等の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
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
    "daily_quotes": [
        {
            "Date": "2023-03-24",
            "Code": "86970",
            "Open": 2047.0,
            "High": 2069.0,
            "Low": 2035.0,
            "Close": 2045.0,
            "UpperLimit": "0",
            "LowerLimit": "0",
            "Volume": 2202500.0,
            "TurnoverValue": 4507051850.0,
            "AdjustmentFactor": 1.0,
            "AdjustmentOpen": 2047.0,
            "AdjustmentHigh": 2069.0,
            "AdjustmentLow": 2035.0,
            "AdjustmentClose": 2045.0,
            "AdjustmentVolume": 2202500.0,
            "MorningOpen": 2047.0,
            "MorningHigh": 2069.0,
            "MorningLow": 2040.0,
            "MorningClose": 2045.5,
            "MorningUpperLimit": "0",
            "MorningLowerLimit": "0",
            "MorningVolume": 1121200.0,
            "MorningTurnoverValue": 2297525850.0,
            "MorningAdjustmentOpen": 2047.0,
            "MorningAdjustmentHigh": 2069.0,
            "MorningAdjustmentLow": 2040.0,
            "MorningAdjustmentClose": 2045.5,
            "MorningAdjustmentVolume": 1121200.0,
            "AfternoonOpen": 2047.0,
            "AfternoonHigh": 2047.0,
            "AfternoonLow": 2035.0,
            "AfternoonClose": 2045.0,
            "AfternoonUpperLimit": "0",
            "AfternoonLowerLimit": "0",
            "AfternoonVolume": 1081300.0,
            "AfternoonTurnoverValue": 2209526000.0,
            "AfternoonAdjustmentOpen": 2047.0,
            "AfternoonAdjustmentHigh": 2047.0,
            "AfternoonAdjustmentLow": 2035.0,
            "AfternoonAdjustmentClose": 2045.0,
            "AfternoonAdjustmentVolume": 1081300.0
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

<table><thead><tr><th>変数名</th><th>説明</th><th width="109">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>Open</td><td>始値（調整前）</td><td>Number</td><td></td></tr><tr><td>High</td><td>高値（調整前）</td><td>Number</td><td></td></tr><tr><td>Low</td><td>安値（調整前）</td><td>Number</td><td></td></tr><tr><td>Close</td><td>終値（調整前）</td><td>Number</td><td></td></tr><tr><td>UpperLimit</td><td>日通ストップ高を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ高以外<br>1：ストップ高</td></tr><tr><td>LowerLimit</td><td>日通ストップ安を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ安以外<br>1：ストップ安</td></tr><tr><td>Volume</td><td>取引高（調整前）</td><td>Number</td><td></td></tr><tr><td>TurnoverValue</td><td>取引代金</td><td>Number</td><td></td></tr><tr><td>AdjustmentFactor</td><td>調整係数</td><td>Number</td><td>株式分割1:2の場合、権利落ち日のレコードにおいて本項目に”0 .5”が収録されます。</td></tr><tr><td>AdjustmentOpen</td><td>調整済み始値</td><td>Number</td><td>※1</td></tr><tr><td>AdjustmentHigh</td><td>調整済み高値</td><td>Number</td><td>※1</td></tr><tr><td>AdjustmentLow</td><td>調整済み安値</td><td>Number</td><td>※1</td></tr><tr><td>AdjustmentClose</td><td>調整済み終値</td><td>Number</td><td>※1</td></tr><tr><td>AdjustmentVolume</td><td>調整済み取引高</td><td>Number</td><td>※1</td></tr><tr><td>MorningOpen</td><td>前場始値</td><td>Number</td><td>※2</td></tr><tr><td>MorningHigh</td><td>前場高値</td><td>Number</td><td>※2</td></tr><tr><td>MorningLow</td><td>前場安値</td><td>Number</td><td>※2</td></tr><tr><td>MorningClose</td><td>前場終値</td><td>Number</td><td>※2</td></tr><tr><td>MorningUpperLimit</td><td>前場ストップ高を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ高以外<br>1：ストップ高<br>※2</td></tr><tr><td>MorningLowerLimit</td><td>前場ストップ安を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ安以外<br>1：ストップ安<br>※2</td></tr><tr><td>MorningVolume</td><td>前場売買高</td><td>Number</td><td>※2</td></tr><tr><td>MorningTurnoverValue</td><td>前場取引代金</td><td>Number</td><td>※2</td></tr><tr><td>MorningAdjustmentOpen</td><td>調整済み前場始値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>MorningAdjustmentHigh</td><td>調整済み前場高値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>MorningAdjustmentLow</td><td>調整済み前場安値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>MorningAdjustmentClose</td><td>調整済み前場終値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>MorningAdjustmentVolume</td><td>調整済み前場売買高</td><td>Number</td><td>※1, ※2</td></tr><tr><td>AfternoonOpen</td><td>後場始値</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonHigh</td><td>後場高値</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonLow</td><td>後場安値</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonClose</td><td>後場終値</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonUpperLimit</td><td>後場ストップ高を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ高以外<br>1：ストップ高<br>※2</td></tr><tr><td>AfternoonLowerLimit</td><td>後場ストップ安を記録したか、否かを表すフラグ</td><td>String</td><td>0：ストップ安以外<br>1：ストップ安<br>※2</td></tr><tr><td>AfternoonVolume</td><td>後場売買高</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonTurnoverValue</td><td>後場取引代金</td><td>Number</td><td>※2</td></tr><tr><td>AfternoonAdjustmentOpen</td><td>調整済み後場始値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>AfternoonAdjustmentHigh</td><td>調整済み後場高値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>AfternoonAdjustmentLow</td><td>調整済み後場安値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>AfternoonAdjustmentClose</td><td>調整済み後場終値</td><td>Number</td><td>※1, ※2</td></tr><tr><td>AfternoonAdjustmentVolume</td><td>調整済み後場売買高</td><td>Number</td><td>※1, ※2</td></tr></tbody></table>

※1 過去の分割等を考慮した調整済みの項目です\
※2 Premiumプランのみ取得可能な項目です

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/prices/daily_quotes?code=86970&date=20230324 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/prices/daily_quotes?code=86970&date=20230324", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
