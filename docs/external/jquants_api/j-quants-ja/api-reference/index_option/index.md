# 日経225オプション四本値(/option/index\_option)

### APIの概要

日経225オプションに関する、四本値や清算値段、理論価格に関する情報を取得することができます。\
また、本APIで取得可能なデータは日経225指数オプション（Weeklyオプション及びフレックスオプションを除く）のみとなります。

### 本APIの留意点

{% hint style="info" %}
**取引セッションについて**

* 2011年2月10日以前は、ナイトセッション、前場、後場で構成されています。
* この期間の前場データは収録されず、後場データが日中場データとして収録されます。なお、日通しデータについては、全立会を含めたデータとなります。
* 2011年2月14日以降は、ナイトセッション、日中場で構成されています。

**レスポンスのキー項目について**

* 緊急取引証拠金が発動した場合は、同一の取引日・銘柄に対して清算価格算出時と緊急取引証拠金算出時のデータが発生します。そのため、Date、Codeに加えてEmergencyMarginTriggerDivisionを組み合わせることでデータを一意に識別することが可能です。
  {% endhint %}

### パラメータ及びレスポンス

## 日次の日経225オプションデータ取得

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/option/index_option`

日付 （date）の指定が必須です。

\*は必須項目

#### Query Parameters

| Name                                   | Type   | Description                                                           |
| -------------------------------------- | ------ | --------------------------------------------------------------------- |
| date<mark style="color:red;">\*</mark> | String | <p>dateの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                    |
| pagination\_key                        | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "index_option": [
        {
            "Date": "2023-03-22",
            "Code": "130060018",
            "WholeDayOpen": 0.0,
            "WholeDayHigh": 0.0,
            "WholeDayLow": 0.0,
            "WholeDayClose": 0.0,
            "NightSessionOpen": 0.0,
            "NightSessionHigh": 0.0,
            "NightSessionLow": 0.0,
            "NightSessionClose": 0.0,
            "DaySessionOpen": 0.0,
            "DaySessionHigh": 0.0,
            "DaySessionLow": 0.0,
            "DaySessionClose": 0.0,
            "Volume": 0.0,
            "OpenInterest": 330.0,
            "TurnoverValue": 0.0,
            "ContractMonth": "2025-06",
            "StrikePrice": 20000.0,
            "Volume(OnlyAuction)": 0.0,
            "EmergencyMarginTriggerDivision": "002",
            "PutCallDivision": "1",
            "LastTradingDay": "2025-06-12",
            "SpecialQuotationDay": "2025-06-13",
            "SettlementPrice": 980.0,
            "TheoreticalPrice": 974.641,
            "BaseVolatility": 17.93025,
            "UnderlyingPrice": 27466.61,
            "ImpliedVolatility": 23.1816,
            "InterestRate": 0.2336
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

| 変数名                            | 説明            | 型             | 備考                                                                            |
| ------------------------------ | ------------- | ------------- | ----------------------------------------------------------------------------- |
| Date                           | 取引日           | String        | YYYY-MM-DD                                                                    |
| Code                           | 銘柄コード         | String        |                                                                               |
| WholeDayOpen                   | 日通し始値         | Number        |                                                                               |
| WholeDayHigh                   | 日通し高値         | Number        |                                                                               |
| WholeDayLow                    | 日通し安値         | Number        |                                                                               |
| WholeDayClose                  | 日通し終値         | Number        |                                                                               |
| NightSessionOpen               | ナイト・セッション始値   | Number/String | 取引開始日初日の銘柄はナイト・セッションが存在しないため、空文字を設定。                                          |
| NightSessionHigh               | ナイト・セッション高値   | Number/String | 同上                                                                            |
| NightSessionLow                | ナイト・セッション安値   | Number/String | 同上                                                                            |
| NightSessionClose              | ナイト・セッション終値   | Number/String | 同上                                                                            |
| DaySessionOpen                 | 日中始値          | Number        |                                                                               |
| DaySessionHigh                 | 日中高値          | Number        |                                                                               |
| DaySessionLow                  | 日中安値          | Number        |                                                                               |
| DaySessionClose                | 日中終値          | Number        |                                                                               |
| Volume                         | 取引高           | Number        |                                                                               |
| OpenInterest                   | 建玉            | Number        |                                                                               |
| TurnoverValue                  | 取引代金          | Number        |                                                                               |
| ContractMonth                  | 限月            | String        | YYYY-MM                                                                       |
| StrikePrice                    | 権利行使価格        | Number        |                                                                               |
| Volume(OnlyAuction)            | 立会内取引高        | Number        | \*1                                                                           |
| EmergencyMarginTriggerDivision | 緊急取引証拠金発動区分   | String        | <p>001: 緊急取引証拠金発動時、002: 清算価格算出時。<br>”001”は2016年7月19日以降に緊急取引証拠金発動した場合のみ収録。</p> |
| PutCallDivision                | プットコール区分      | String        | 1: プット、2: コール                                                                 |
| LastTradingDay                 | 取引最終年月日       | String        | YYYY-MM-DD \*1                                                                |
| SpecialQuotationDay            | sq日           | String        | YYYY-MM-DD \*1                                                                |
| SettlementPrice                | 清算値段          | Number        | \*1                                                                           |
| TheoreticalPrice               | 理論価格          | Number        | \*1                                                                           |
| BaseVolatility                 | 基準ボラティリティ     | Number        | アット・ザ・マネープット及びコールそれぞれのインプライドボラティリティの中間値 \*1                                   |
| UnderlyingPrice                | 原証券価格         | Number        | \*1                                                                           |
| ImpliedVolatility              | インプライドボラティリティ | Number        | \*1                                                                           |
| InterestRate                   | 理論価格計算用金利     | Number        | \*1                                                                           |

\*1 2016年7月19日以降のみ提供。

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/option/index_option?date=20230324 -H "Authorization: Bearer $idToken" 
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
r = requests.get("https://api.jquants.com/v1/option/index_option?date=20230324", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
