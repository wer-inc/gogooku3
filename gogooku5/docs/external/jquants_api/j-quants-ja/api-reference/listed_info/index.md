# 上場銘柄一覧(/listed/info)

### API概要

過去時点での銘柄情報、当日の銘柄情報および翌営業日時点の銘柄情報が取得可能です。\
ただし、翌営業日時点の銘柄情報については17 時半以降に取得可能となります。

### 本APIの留意点

{% hint style="info" %}
過去日付の指定について、Premiumプランでデータ提供開始日（2008年5月7日）より過去日付を指定した場合であっても、2008年5月7日時点の銘柄情報を返却します。\
また、指定された日付が休業日の場合は指定日の翌営業日の銘柄情報を返却します。

貸借信用区分および貸借信用区分名については StandardおよびPremiumプランのみ取得可能な項目です。
{% endhint %}

{% hint style="warning" %}
2022年4月の東証市場区分再編により、日本銀行（銘柄コード83010）および信金中央金庫（銘柄コード84210）については、制度上所属する市場区分が存在しなくなりましたが、J-Quantsでは市場区分をスタンダードとして返却しています。
{% endhint %}

データの取得では、銘柄コード（code）または日付（date）の指定が可能です。\
各パラメータの組み合わせとレスポンスの結果については以下のとおりです。

<table><thead><tr><th width="126" data-type="checkbox">code</th><th width="133" data-type="checkbox">date</th><th>レスポンスの結果</th></tr></thead><tbody><tr><td>false</td><td>false</td><td>APIを実行した日付時点における全銘柄情報一覧 *1</td></tr><tr><td>true</td><td>false</td><td>APIを実行した日付時点における指定された銘柄情報 *1</td></tr><tr><td>false</td><td>true</td><td>指定日付時点における全銘柄情報の一覧 *2</td></tr><tr><td>true</td><td>true</td><td>指定日付時点における指定された銘柄情報 *2</td></tr></tbody></table>

\*1 休業日において日付を指定せずにクエリした場合、直近の翌営業日における銘柄情報一覧を返却します。\
\*2 未来日付の指定について、Lightプラン以上では翌営業日時点のデータが取得可能です。翌営業日より先の未来日付を指定した場合であっても、翌営業日時点の銘柄情報を返却します。\\

## 日次の銘柄情報を取得します

<mark style="color:blue;">`GET`</mark> `https://api.jquants.com/v1/listed/info`

\*は必須項目

#### Query Parameters

| Name | Type   | Description                                                                              |
| ---- | ------ | ---------------------------------------------------------------------------------------- |
| code | String | <p>27890 or 2789</p><p>4桁の銘柄コードを指定した場合は、普通株式と優先株式の両方が上場している銘柄においては普通株式のデータのみが取得されます。</p> |
| date | String | <p>基準なる日付の指定</p><p>（e.g. 20210907 or 2021-09-07）</p>                                     |

#### Headers

| Name                                            | Type   | Description |
| ----------------------------------------------- | ------ | ----------- |
| Authorization<mark style="color:red;">\*</mark> | String | アクセスキー      |

{% tabs %}
{% tab title="200: OK " %}

```json
{
    "info": [
        {
            "Date": "2022-11-11",
            "Code": "86970",
            "CompanyName": "日本取引所グループ",
        　　　　　　　　"CompanyNameEnglish": "Japan Exchange Group,Inc.",
            "Sector17Code": "16",
            "Sector17CodeName": "金融（除く銀行）",
            "Sector33Code": "7200",
            "Sector33CodeName": "その他金融業",
            "ScaleCategory": "TOPIX Large70",
            "MarketCode": "0111",
            "MarketCodeName": "プライム",
            "MarginCode": "1",
            "MarginCodeName": "信用",
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

<table><thead><tr><th width="231">変数名</th><th>説明</th><th width="97">型</th><th>備考</th></tr></thead><tbody><tr><td>Date</td><td>情報適用年月日</td><td>String</td><td>YYYY-MM-DD</td></tr><tr><td>Code</td><td>銘柄コード</td><td>String</td><td></td></tr><tr><td>CompanyName</td><td>会社名</td><td>String</td><td></td></tr><tr><td>CompanyNameEnglish</td><td>会社名（英語）</td><td>String</td><td></td></tr><tr><td>Sector17Code</td><td>17業種コード</td><td>String</td><td><a href="listed_info/sector17code">17業種コード及び業種名</a>を参照</td></tr><tr><td>Sector17CodeName</td><td>17業種コード名</td><td>String</td><td><a href="listed_info/sector17code">17業種コード及び業種名</a>を参照</td></tr><tr><td>Sector33Code</td><td>33業種コード</td><td>String</td><td><a href="listed_info/sector33code">33業種コード及び業種名</a>を参照</td></tr><tr><td>Sector33CodeName</td><td>33業種コード名</td><td>String</td><td><a href="listed_info/sector33code">33業種コード及び業種名</a>を参照</td></tr><tr><td>ScaleCategory</td><td>規模コード</td><td>String</td><td></td></tr><tr><td>MarketCode</td><td>市場区分コード</td><td>String</td><td><a href="listed_info/marketcode">市場区分コード及び市場区分</a>を参照</td></tr><tr><td>MarketCodeName</td><td>市場区分名</td><td>String</td><td><a href="listed_info/marketcode">市場区分コード及び市場区分</a>を参照</td></tr><tr><td>MarginCode</td><td>貸借信用区分</td><td>String</td><td>1: 信用、2: 貸借、3: その他<br>※1</td></tr><tr><td>MarginCodeName</td><td>貸借信用区分名</td><td>String</td><td>※1</td></tr></tbody></table>

※1 StandardおよびPremiumプランで取得可能な項目です

### APIコールサンプルコード

{% tabs %}
{% tab title="Curl" %}
{% code overflow="wrap" %}

```bash
idToken=<YOUR idToken> && curl https://api.jquants.com/v1/listed/info -H "Authorization: Bearer $idToken"
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
r = requests.get("https://api.jquants.com/v1/listed/info", headers=headers)
r.json()
```

{% endcode %}
{% endtab %}
{% endtabs %}
