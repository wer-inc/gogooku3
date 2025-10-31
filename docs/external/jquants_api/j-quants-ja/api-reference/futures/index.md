# 先物四本値(/derivatives/futures)

### APIの概要

先物に関する、四本値や清算値段、理論価格に関する情報を取得することができます。\
また、本APIで取得可能なデータについては[先物商品区分コード一覧](https://jpx.gitbook.io/j-quants-ja/api-reference/futures/derivativeproductcategory)を参照ください。

### 本APIの留意点

{% hint style="info" %}
**銘柄コードについて**

* 先物・オプション取引識別コードの付番規則については[証券コード関係の関係資料等](https://www.jpx.co.jp/sicc/securities-code/01.html)を参照してください。

**取引セッションについて**

* 2011年2月10日以前は、ナイトセッション、前場、後場で構成されています。
* この期間の前場データは収録されず、後場データが日中場データとして収録されます。なお、日通しデータについては、全立会を含めたデータとなります。
* 2011年2月14日以降は、ナイトセッション、日中場で構成されています。

**祝日取引について**

* 祝日取引の取引日については、祝日取引実施日直前の平日に開始するナイト・セッション（祝日前営業日）及び祝日取引実施日直後の平日（祝日翌営業日）のデイ・セッションと同一の取引日として扱います。

**レスポンスのキー項目について**

* 緊急取引証拠金が発動した場合は、同一の取引日・銘柄に対して清算価格算出時と緊急取引証拠金算出時のデータが発生します。そのため、Date、Codeに加えてEmergencyMarginTriggerDivisionを組み合わせることでデータを一意に識別することが可能です。
  {% endhint %}

### パラメータ及びレスポンス

## 日次の先物四本値データ取得

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/derivatives/futures`

データの取得では、日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name                                   | Type   | Description                                                           |
| -------------------------------------- | ------ | --------------------------------------------------------------------- |
| category                               | String | 商品区分の指定                                                               |
| date<mark style="color:red;">\*</mark> | String | <p>dateの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                    |
| contract\_flag                         | String | 中心限月フラグの指定                                                            |
| pagination\_key                        | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "futures": [
        {
             "Code": "169090005",
             "DerivativesProductCategory": "TOPIXF",
             "Date": "2024-07-23", 
             "WholeDayOpen": 2825.5, 
             "WholeDayHigh": 2853.0, 
             "WholeDayLow": 2825.5, 
             "WholeDayClose": 2829.0, 
             "MorningSessionOpen": "", 
             "MorningSessionHigh": "", 
             "MorningSessionLow": "", 
             "MorningSessionClose": "", 
             "NightSessionOpen": 2825.5, 
             "NightSessionHigh": 2850.0, 
             "NightSessionLow": 2825.5, 
             "NightSessionClose": 2845.0, 
             "DaySessionOpen": 2850.5, 
             "DaySessionHigh": 2853.0, 
             "DaySessionLow": 2826.0, 
             "DaySessionClose": 2829.0, 
             "Volume": 42910.0, 
             "OpenInterest": 479812.0, 
             "TurnoverValue": 1217918971856.0, 
             "ContractMonth": "2024-09", 
             "Volume(OnlyAuction)": 40405.0, 
             "EmergencyMarginTriggerDivision": "002", 
             "LastTradingDay": "2024-09-12", 
             "SpecialQuotationDay": "2024-09-13", 
             "SettlementPrice": 2829.0, 
             "CentralContractMonthFlag": "1"
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

| 変数名                            | 説明          | 型             | 備考                                                                            |
| ------------------------------ | ----------- | ------------- | ----------------------------------------------------------------------------- |
| Code                           | 銘柄コード       | String        |                                                                               |
| DerivativeProductCategory      | 先物商品区分      | String        |                                                                               |
| Date                           | 取引日         | String        | YYYY-MM-DD                                                                    |
| WholeDayOpen                   | 日通し始値       | Number        |                                                                               |
| WholeDayHigh                   | 日通し高値       | Number        |                                                                               |
| WholeDayLow                    | 日通し安値       | Number        |                                                                               |
| WholeDayClose                  | 日通し終値       | Number        |                                                                               |
| MorningSessionOpen             | 前場始値        | Number/String | 前後場取引対象銘柄でない場合、空文字を設定。                                                        |
| MorningSessionHigh             | 前場高値        | Number/String | 同上                                                                            |
| MorningSessionLow              | 前場安値        | Number/String | 同上                                                                            |
| MorningSessionClose            | 前場終値        | Number/String | 同上                                                                            |
| NightSessionOpen               | ナイト・セッション始値 | Number/String | 取引開始日初日の銘柄はナイト・セッションが存在しないため、空文字を設定。                                          |
| NightSessionHigh               | ナイト・セッション高値 | Number/String | 同上                                                                            |
| NightSessionLow                | ナイト・セッション安値 | Number/String | 同上                                                                            |
| NightSessionClose              | ナイト・セッション終値 | Number/String | 同上                                                                            |
| DaySessionOpen                 | 日中始値        | Number        |                                                                               |
| DaySessionHigh                 | 日中高値        | Number        |                                                                               |
| DaySessionLow                  | 日中安値        | Number        |                                                                               |
| DaySessionClose                | 日中終値        | Number        |                                                                               |
| Volume                         | 取引高         | Number        |                                                                               |
| OpenInterest                   | 建玉          | Number        |                                                                               |
| TurnoverValue                  | 取引代金        | Number        |                                                                               |
| ContractMonth                  | 限月          | String        | YYYY-MM                                                                       |
| Volume(OnlyAuction)            | 立会内取引高      | Number        | \*1                                                                           |
| EmergencyMarginTriggerDivision | 緊急取引証拠金発動区分 | String        | <p>001: 緊急取引証拠金発動時、002: 清算価格算出時。<br>”001”は2016年7月19日以降に緊急取引証拠金発動した場合のみ収録。</p> |
| LastTradingDay                 | 取引最終年月日     | String        | YYYY-MM-DD \*1                                                                |
| SpecialQuotationDay            | sq日         | String        | YYYY-MM-DD \*1                                                                |
| SettlementPrice                | 清算値段        | Number        | \*1                                                                           |
| CentralContractMonthFlag       | 中心限月フラグ     | String        | 1:中心限月、0:その他 \*1                                                              |

\*1 2016年7月19日以降のみ提供。

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/derivatives/futures?date=20230324 -H "Authorization: Bearer $idToken" 
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
r = requests.get("https://api.jquants.com/v1/derivatives/futures?date=20230324", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
