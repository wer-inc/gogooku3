# API共通の留意事項

### レートリミットについて

APIにはレートリミットが設定されております。一定の使用量を超えますと、一時的にAPIの利用が制限されます。本制限の基準は皆様のご利用状況を確認した上で、調整をいたします。

### レスポンスのページングについて

APIのレスポンスが大容量になった場合、レスポンスに`pagination_key`が設定される場合があります。`pagination_key`が設定された場合、次のクエリにおいて検索条件を変更せずに`pagination_key`を設定してリクエストを実行することで後続のデータを取得することが可能です。レスポンスの形式は各APIのサンプルコードを参照ください。

`pagination_key`を利用したPythonのサンプルコードを以下に記載します。

```python
import requests
import json

# 通常の検索
idToken = "YOUR idToken"
headers = {"Authorization": "Bearer {}".format(idToken)}
r_get = requests.get("https://api.jquants.com/v1/method?query=param", headers=headers)
data = r_get.json()["data"]

# 大容量データが返却された場合の再検索
# データ量により複数ページ取得できる場合があるため、pagination_keyが含まれる限り、再検索を実施
while "pagination_key" in r_get.json():
    pagination_key = r_get.json()["pagination_key"]
    r_get = requests.get(f"https://api.jquants.com/v1/method?query=param&pagination_key={pagination_key}", headers=headers)
    data += r_get.json()["data"]

```

{% hint style="info" %}

* クエリに対する全ての該当データを返却するまで、`pagination_key`がレスポンスメッセージに設定されます。`pagination_key`がレスポンスメッセージに設定されない場合はクエリに対する全ての該当データが返却されたことを意味します。
* ページングの都度、`pagination_key`の値は変わります。
  {% endhint %}

### APIレスポンスのGzip化について <a href="#gzip" id="gzip"></a>

データ通信量削減を目的としてAPIからのレスポンスをGzip化しています。

本仕様変更に伴うクライアント側での対処は基本的に不要であることを想定しておりますが、利用パターンによってはクライアント側での対処が必要となります。具体的な対処の要否につきましては以下の表をご参照ください。

#### ユーザの利用パターンごとの影響有無

<table data-full-width="false"><thead><tr><th width="205">パッケージ*使用有無</th><th width="242">Accept-Encoding:gzip 有無</th><th>クライアント側での対処の要否</th></tr></thead><tbody><tr><td>パッケージ使用</td><td>デフォルトで上記ヘッダーが付与</td><td>対処不要<br>(圧縮されたレスポンスが自動的に解凍されるためクライアント側での考慮は不要)</td></tr><tr><td>パッケージ不使用</td><td>上記headerあり</td><td><mark style="color:red;">圧縮されたレスポンスの適切な解凍処理が必要</mark><br>(curlの場合 <code>--compressed</code>)</td></tr><tr><td></td><td>上記headerなし</td><td>対処不要<br>(未圧縮のレスポンスを受信するためクライアント側での考慮は不要)</td></tr></tbody></table>

&#x20;\* 一般的にRestAPIコール時に利用されるHTTPクライアントライブラリのことを指します。\
例) pythonにおけるrequestsやurllib等のライブラリ
