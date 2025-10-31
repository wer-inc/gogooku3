# 取引カレンダー(/markets/trading\_calendar)

### 取引カレンダーAPI概要

東証およびOSEにおける営業日、休業日、ならびにOSEにおける祝日取引の有無の情報を取得できます。\
配信データは下記のページで公表している内容と同様です。\
\
休業日一覧\
<https://www.jpx.co.jp/corporate/about-jpx/calendar/index.html>[\
](https://www.jpx.co.jp/derivatives/rules/holidaytrading/index.html)祝日取引実施日\
<https://www.jpx.co.jp/derivatives/rules/holidaytrading/index.html>

### 本APIの留意点

{% hint style="info" %}

* 原則として、毎年3月末頃をめどに翌年1年間の営業日および祝日取引実施日（予定）を更新します。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、休日区分（holidaydivision）または日付（from/to）の指定が可能です。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="163" data-type="checkbox">holidaydivision</th><th width="110" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>指定された休日区分について全期間分のデータ</td></tr><tr><td>true</td><td>true</td><td>指定された休日区分について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>指定された期間分のデータ</td></tr><tr><td>false</td><td>false</td><td>全期間分のデータ</td></tr></tbody></table>

休日区分（holidaydivision）で指定可能なパラメータについてはこちらを参照ください。

{% content-ref url="trading\_calendar/holiday\_division" %}
[holiday\_division](https://jpx.gitbook.io/j-quants-ja/api-reference/trading_calendar/holiday_division)
{% endcontent-ref %}

## 営業日のデータを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/markets/trading_calendar`

データの取得では、休日区分（holidaydivision）または日付（from/to）の指定が可能です。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                        |
| --------------- | ------ | -------------------------------------------------- |
| holidaydivision | String | 休日区分                                               |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p> |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>   |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "trading_calendar": [
        {
            "Date": "2015-04-01",
            "HolidayDivision": "1"
        }
    ]
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

<table><thead><tr><th width="172">変数名</th><th width="243">説明</th><th width="102">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>HolidayDivision</td><td>休日区分</td><td>String</td><td><a href="trading_calendar/holiday_division">休日区分</a>を参照</td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/markets/trading_calendar?holidaydivision=1&from=20220101&to=20221231 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/markets/trading_calendar?holidaydivision=1&from=20220101&to=20221231", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
