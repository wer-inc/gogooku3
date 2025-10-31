# API利用までの流れ

## 全体の流れ

* APIを利用するためには[ランディングページ](https://jpx-jquants.com/)からユーザ登録およびサブスクリプションプランの登録を行います（初回のみ実施）。
* その後、リフレッシュトークンおよびIDトークンの取得を行ってください（API利用の都度実施）。
* IDトークン取得後、取得したIDトークンを用いて各APIをご利用ください。

### クイックスタートガイド

Google Colaboratoryを利用したJ-Quants APIのクイックスタートガイドがございます。以下の「Open in Colab」リンクよりお試しください。

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-size="line">](https://colab.research.google.com/github/J-Quants/jquants-api-quick-start/blob/master/jquants-api-quick-start.ipynb)

### Step1：API利用開始までの流れ　※初回のみ実施

1. [ユーザ登録ページ](https://jpx-jquants.com/auth/signup/)から、J-Quants APIのサービスにメールアドレスを仮登録します。
2. 仮登録したメールアドレスに送られるURLをクリックしユーザ登録を完了させます。
3. [サインインページ](https://jpx-jquants.com/auth/signin)からメニュー画面にログイン後、サブスクリプションプランの登録を行います。

{% hint style="warning" %}

* API利用するためにはFreeプランも含めたいずれかのサブスクリプションプランへの登録が必要です。ユーザ登録とサブスクリプションプランの違いについては[FAQ](https://jpx.gitbook.io/j-quants-ja/faq/plan)を参照ください。
  {% endhint %}

### Step2：リフレッシュトークン取得の流れ　※API利用の都度実施

1. [メニューページ](https://jpx-jquants.com/dashboard/menu/)にログイン後、リフレッシュトークンをコピーします。\
   また、Web画面とは別に下記のAPIによりリフレッシュトークンを取得することも可能です。

{% content-ref url="../api-reference/refreshtoken" %}
[refreshtoken](https://jpx.gitbook.io/j-quants-ja/api-reference/refreshtoken)
{% endcontent-ref %}

{% hint style="warning" %}

* リフレッシュトークンは１週間で失効しますので、失効後はメニューページにログインするか、またはリフレッシュトークン取得APIを使用し、再度リフレッシュトークンを取得してください。
* 有償版とBeta版ではAPIの接続先ドメイン名が異なりますのでご留意ください。\
  以下はリフレッシュトークン取得APIの例です。

  　<mark style="color:red;">**有償版**</mark>　[<mark style="color:red;">**https://api.jquants.com/v1/token/auth\_user**</mark>](https://api.jquants.com/v1/token/auth_user)\
  　Beta版　<https://api.jpx-jquants.com/v1/token/auth\\_user>
  {% endhint %}

### Step3：IDトークンの取得　※API利用の都度実施

APIで各データを取得するためにはIDトークンが必要になります。上記で取得したリフレッシュトークンを、下記のIDトークン取得APIにセットして、IDトークンを取得します。

{% content-ref url="../api-reference/idtoken" %}
[idtoken](https://jpx.gitbook.io/j-quants-ja/api-reference/idtoken)
{% endcontent-ref %}

{% hint style="warning" %}
IDトークンは24時間で失効します。

失効後はIDトークン取得APIを用いて再度取得をお願いします。
{% endhint %}

### Step4：取得したIDトークンを用いて各APIをご利用ください。

具体的な各APIの仕様は以下を参照ください。

{% content-ref url="../api-reference" %}
[api-reference](https://jpx.gitbook.io/j-quants-ja/api-reference)
{% endcontent-ref %}
