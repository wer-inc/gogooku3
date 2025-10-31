# 決算発表予定日(/fins/announcement)

### API概要

3月期・9月期決算の会社の決算発表予定日を取得できます。（その他の決算期の会社は今後対応予定です）

### 本APIの留意点

{% hint style="info" %}

* 下記のサイトで、3月期・９月期決算会社分に更新があった場合のみ19時ごろに更新されます。３月期・９月期決算会社についての更新がなかった場合は、最終更新日時点のデータを提供します。\
  <https://www.jpx.co.jp/listing/event-schedules/financial-announcement/index.html>
* 本APIは翌営業日に決算発表が行われる銘柄に関する情報を返します。
* 本APIから得られたデータにおいてDateの項目が翌営業日付であるレコードが存在しない場合は、3月期・9月期決算会社における翌営業日の開示予定はないことを意味します。
* REITのデータは含まれません。
  {% endhint %}

### パラメータ及びレスポンス

## 決算発表予定日の銘柄コード、年度、 四半期等の照会をします。

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/fins/announcement`

\*は必須項目

#### Query Parameters

| Name            | Type   | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| pagination\_key | String | <p>検索の先頭を指定する文字列</p><p>過去の検索で返却された<code>pagination\_key</code>を設定</p> |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```javascript
{
  "announcement": [
    {
      "Date": "2022-02-14",
      "Code": "43760",
      "CompanyName": "くふうカンパニー",
      "FiscalYear": "9月30日",
      "SectorName": "情報・通信業",
      "FiscalQuarter": "第１四半期",
      "Section": "マザーズ"
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
{% endtabs %}

### データ項目概要

<table><thead><tr><th>変数名</th><th>説明</th><th width="107">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>日付</td><td>String</td><td>例：YYYY-MM-DD<br>なお、決算発表予定日が未定の場合、空文字("")となります。</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>CompanyName</td><td>会社名</td><td>String</td><td></td></tr><tr><td>FiscalYear</td><td>決算期末</td><td>String</td><td></td></tr><tr><td>SectorName</td><td>業種名</td><td>String</td><td></td></tr><tr><td>FiscalQuarter</td><td>決算種別</td><td>String</td><td></td></tr><tr><td>Section</td><td>市場区分</td><td>String</td><td></td></tr></tbody></table>

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/fins/announcement -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/fins/announcement", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
