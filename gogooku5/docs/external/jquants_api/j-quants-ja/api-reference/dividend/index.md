# 配当金情報(/fins/dividend)

### API概要

上場会社の配当（決定・予想）に関する１株当たり配当金額、基準日、権利落日及び支払開始予定日等の情報を取得できます。

### 本APIの留意点

{% hint style="info" %}

* 東証上場銘柄でない銘柄（地方取引所単独上場銘柄）についてはデータの収録対象外となっております。
  {% endhint %}

### パラメータ及びレスポンス

データの取得では、銘柄コード（code）または通知日付（date）の指定が必須となります。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="109" data-type="checkbox">code</th><th width="93" data-type="checkbox">date</th><th width="103" data-type="checkbox">from /to</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>true</td><td>false</td><td>false</td><td>指定された銘柄について取得可能期間の全データ</td></tr><tr><td>true</td><td>false</td><td>true</td><td>指定された銘柄について指定された期間分のデータ</td></tr><tr><td>false</td><td>true</td><td>false</td><td>全上場銘柄について指定された通知日付のデータ</td></tr></tbody></table>

## 配当金データを取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/fins/dividend`

データの取得では、銘柄コード（code）または通知日付（date）の指定が必須となります。

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| code            | String | <p>銘柄コード</p><p>（e.g. 27800 or 2780）</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| from            | String | <p>fromの指定</p><p>（e.g. 20210901 or 2021-09-01）</p>                                                          |
| to              | String | <p>toの指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                            |
| date            | String | <p>\*fromとtoを指定しないとき</p><p>（e.g. 20210907 or 2021-09-07）</p>                                                |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p>                                       |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

<pre class="language-json"><code class="lang-json"><strong>{
</strong>    "dividend": [
        {
            "AnnouncementDate": "2014-02-24",
            "AnnouncementTime": "09:21",
            "Code": "15550",
            "ReferenceNumber": "201402241B00002",
            "StatusCode": "1",
            "BoardMeetingDate": "2014-02-24",
            "InterimFinalCode": "2",
            "ForecastResultCode": "2",
            "InterimFinalTerm": "2014-03",
            "GrossDividendRate": "-",
            "RecordDate": "2014-03-10",
            "ExDate": "2014-03-06",
            "ActualRecordDate": "2014-03-10",
            "PayableDate": "-",
            "CAReferenceNumber": "201402241B00002",
            "DistributionAmount": "",
            "RetainedEarnings": "",
            "DeemedDividend": "",
            "DeemedCapitalGains": "",
            "NetAssetDecreaseRatio": "",
            "CommemorativeSpecialCode": "0",
            "CommemorativeDividendRate": "",
            "SpecialDividendRate": ""
        }
    ],
    "pagination_key": "value1.value2."
}
</code></pre>

{% endtab %}

{% tab title="400: Bad Request " %}

```javascript
{
    "message": <Error Message>
}
```

{% endtab %}

{% tab title="401: Unauthorized " %}

```javascript
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

<table><thead><tr><th width="214">変数名</th><th width="188">説明</th><th width="100">型</th><th>備考</th></tr></thead><tbody><tr><td>AnnouncementDate</td><td>通知日時（年月日）</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>AnnouncementTime</td><td>通知日時（時分）</td><td>String</td><td>HH:MI</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>ReferenceNumber</td><td>リファレンスナンバー</td><td>String</td><td><p>配当通知を一意に特定するための番号</p><p>※<a href="#reference">リファレンスナンバーについて</a>を参照</p></td></tr><tr><td>StatusCode</td><td>更新区分（コード）</td><td>String</td><td>1: 新規、2: 訂正、3: 削除</td></tr><tr><td>BoardMeetingDate</td><td>取締役会決議日</td><td>String</td><td></td></tr><tr><td>InterimFinalCode</td><td>配当種類（コード）</td><td>String</td><td>1: 中間配当、2: 期末配当</td></tr><tr><td>ForecastResultCode</td><td>予想／決定（コード）</td><td>String</td><td>1: 決定、2: 予想</td></tr><tr><td>InterimFinalTerm</td><td>配当基準日年月</td><td>String</td><td></td></tr><tr><td>GrossDividendRate</td><td>１株当たり配当金額</td><td>Number/String</td><td>未定の場合: - 、非設定の場合: 空文字</td></tr><tr><td>RecordDate</td><td>基準日</td><td>String</td><td></td></tr><tr><td>ExDate</td><td>権利落日</td><td>String</td><td></td></tr><tr><td>ActualRecordDate</td><td>権利確定日</td><td>String</td><td></td></tr><tr><td>PayableDate</td><td>支払開始予定日</td><td>String</td><td>未定の場合: - 、非設定の場合: 空文字</td></tr><tr><td>CAReferenceNumber</td><td>ＣＡリファレンスナンバー</td><td>String</td><td><p>訂正・削除の対象となっている配当通知のリファレンスナンバー。新規の場合はリファレンスナンバーと同じ値を設定</p><p>※<a href="#reference">リファレンスナンバーについて</a>を参照</p></td></tr><tr><td>DistributionAmount</td><td>1株当たりの交付金銭等の額</td><td>Number/String</td><td>未定の場合: - 、非設定の場合: 空文字　が設定されます。<br>2014年2月24日以降のみ提供。</td></tr><tr><td>RetainedEarnings</td><td>1株当たりの利益剰余金の額</td><td>Number/String</td><td>同上</td></tr><tr><td>DeemedDividend</td><td>1株当たりのみなし配当の額</td><td>Number/String</td><td>同上</td></tr><tr><td>DeemedCapitalGains</td><td>1株当たりのみなし譲渡収入の額</td><td>Number/String</td><td>同上</td></tr><tr><td>NetAssetDecreaseRatio</td><td>純資産減少割合</td><td>Number/String</td><td>同上</td></tr><tr><td>CommemorativeSpecialCode</td><td>記念配当/特別配当コード</td><td>String</td><td>1: 記念配当、2: 特別配当、3: 記念・特別配当、0: 通常の配当</td></tr><tr><td>CommemorativeDividendRate</td><td>１株当たり記念配当金額</td><td>Number/String</td><td>未定の場合: - 、非設定の場合: 空文字<br>2022年6月6日以降のみ提供。</td></tr><tr><td>SpecialDividendRate</td><td>１株当たり特別配当金額</td><td>Number/String</td><td>同上</td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/fins/dividend?code=86970 -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/fins/dividend?code=86970", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}

### 付録：リファレンスナンバーについて <a href="#reference" id="reference"></a>

* リファレンスナンバー : 配当通知を一意に特定するための番号。
* CAリファレンスナンバー：訂正・削除の対象となっている配当通知のリファレンスナンバー。新規の場合はリファレンスナンバーと同じ値。

#### 具体例：以下の通知があった場合に、提供データは下表のとおりになります。

* 銘柄：日本取引所（銘柄コード：86970）について
  * 2023-03-06　　配当が新規で通知
  * 2023-03-07　　配当が訂正情報として通知
  * 2023-03-08　　配当が削除された
  * 2023-03-09　　配当が新規で通知

<table><thead><tr><th width="156">AnnouncementDate</th><th width="91">Code</th><th width="179">ReferenceNumber</th><th width="178">CAReferenceNumber</th><th>StatusCode</th></tr></thead><tbody><tr><td>2023-03-06</td><td>86970</td><td>1</td><td>1</td><td>1：新規</td></tr><tr><td>2023-03-07</td><td>86970</td><td>2</td><td>1</td><td>2：訂正</td></tr><tr><td>2023-03-08</td><td>86970</td><td>3</td><td>1</td><td>3：削除</td></tr><tr><td>2023-03-09</td><td>86970</td><td>4</td><td>4</td><td>1：新規</td></tr></tbody></table>

\* 一部項目のみを抽出して例示しています。\
\* 上記のコード値は例示のため便宜的な記載としており、また実際に発生したデータとは異なります。
