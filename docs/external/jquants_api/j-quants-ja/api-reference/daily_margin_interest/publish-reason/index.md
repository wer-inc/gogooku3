# 公表の理由

日々公表銘柄に指定されている理由をフラグを用いて示します。

例えば以下のケースでは、東京証券取引所が定める信用取引の規制措置銘柄と日本証券金融における貸株申込制限措置銘柄に選定されていることを意味します。

`{ Restricted: 1, DailyPublication: 0, Monitoring: 0, RestrictedByJSF: 1, PrecautionByJSF: 0, UnclearOrSecOnAlert: 0}`

<table><thead><tr><th width="219.5">項目</th><th>意味</th></tr></thead><tbody><tr><td>Restricted</td><td>1の場合、<a href="https://www.jpx.co.jp/markets/equities/margin-reg/index.html">東京証券取引所が定める信用取引の規制措置銘柄</a>に選定されている。0の場合、非該当。</td></tr><tr><td>DailyPublication</td><td>1の場合、<a href="https://www.jpx.co.jp/markets/equities/margin-daily/index.html">東京証券取引所が定める日々公表銘柄</a>に選定されている。0の場合、非該当。</td></tr><tr><td>Monitoring</td><td>1の場合、<a href="https://www.jpx.co.jp/listing/measures/alert/index.html">東京証券取引所が定める特別注意銘柄</a>に選定されている。0の場合、非該当。</td></tr><tr><td>RestrictedByJSF</td><td>1の場合、<a href="https://www.taisyaku.jp/brand/">日本証券金融が定める貸株申込制限措置銘柄</a>に選定されている。0の場合、非該当。</td></tr><tr><td>PrecautionByJSF</td><td>1の場合、<a href="https://www.taisyaku.jp/brand/">日本証券金融が定める貸株注意喚起銘柄</a>に選定されている。0の場合、非該当。</td></tr><tr><td>UnclearOrSecOnAlert</td><td>1の場合、<a href="https://www.jpx.co.jp/markets/equities/alerts/index.html">東京証券取引所が定める不明確情報等により注意喚起の対象となった銘柄、特別注意銘柄等</a>に選定されている。0の場合、非該当。</td></tr></tbody></table>
