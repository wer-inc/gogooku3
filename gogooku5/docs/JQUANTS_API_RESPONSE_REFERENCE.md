# J-Quants API Response Schema Reference

Comprehensive reference for all J-Quants API response schemas extracted from official documentation.

**Document Version**: 2025-11-06
**Source**: J-Quants API Documentation (gogooku5/docs/external/jquants_api/j-quants-ja/api-reference/)

---

## Table of Contents

1. [Announcement (決算発表予定)](#1-announcement-決算発表予定)
2. [Breakdown (売買内訳データ)](#2-breakdown-売買内訳データ)
3. [Daily Margin Interest (日々公表信用取引残高)](#3-daily-margin-interest-日々公表信用取引残高)
4. [Daily Quotes (株価四本値)](#4-daily-quotes-株価四本値)
5. [Dividend (配当金情報)](#5-dividend-配当金情報)
6. [Futures (先物四本値)](#6-futures-先物四本値)
7. [Index Option (日経225オプション)](#7-index-option-日経225オプション)
8. [Indices (指数四本値)](#8-indices-指数四本値)
9. [Listed Info (上場銘柄一覧)](#9-listed-info-上場銘柄一覧)
10. [Options (オプション四本値)](#10-options-オプション四本値)
11. [Prices AM (前場四本値)](#11-prices-am-前場四本値)
12. [Short Selling (業種別空売り比率)](#12-short-selling-業種別空売り比率)
13. [Short Selling Positions (空売り残高報告)](#13-short-selling-positions-空売り残高報告)
14. [Statements (財務情報)](#14-statements-財務情報)
15. [TOPIX (TOPIX指数)](#15-topix-topix指数)
16. [Trades Spec (投資部門別情報)](#16-trades-spec-投資部門別情報)
17. [Trading Calendar (取引カレンダー)](#17-trading-calendar-取引カレンダー)
18. [Weekly Margin Interest (信用取引週末残高)](#18-weekly-margin-interest-信用取引週末残高)

---

## 1. Announcement (決算発表予定)

**Endpoint**: `GET /v1/fins/announcement`

**Description**: 3月期・9月期決算の会社の決算発表予定日を取得（その他の決算期は今後対応予定）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD format, 空文字の場合は未定) |
| Code | String | 銘柄コード |
| CompanyName | String | 会社名 |
| FiscalYear | String | 決算期末 |
| SectorName | String | 業種名 |
| FiscalQuarter | String | 決算種別 |
| Section | String | 市場区分 |

### Response Format
```json
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

---

## 2. Breakdown (売買内訳データ)

**Endpoint**: `GET /v1/markets/breakdown`

**Description**: 東証上場銘柄の日次売買代金・売買高（信用取引や空売りフラグ情報で細分化）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 売買日 (YYYY-MM-DD) |
| Code | String | 銘柄コード |
| LongSellValue | Number | 実売りの約定代金 |
| ShortSellWithoutMarginValue | Number | 空売り(信用新規売りを除く)の約定代金 |
| MarginSellNewValue | Number | 信用新規売りの約定代金 |
| MarginSellCloseValue | Number | 信用返済売りの約定代金 |
| LongBuyValue | Number | 現物買いの約定代金 |
| MarginBuyNewValue | Number | 信用新規買いの約定代金 |
| MarginBuyCloseValue | Number | 信用返済買いの約定代金 |
| LongSellVolume | Number | 実売りの約定株数 |
| ShortSellWithoutMarginVolume | Number | 空売り(信用新規売りを除く)の約定株数 |
| MarginSellNewVolume | Number | 信用新規売りの約定株数 |
| MarginSellCloseVolume | Number | 信用返済売りの約定株数 |
| LongBuyVolume | Number | 現物買いの約定株数 |
| MarginBuyNewVolume | Number | 信用新規買いの約定株数 |
| MarginBuyCloseVolume | Number | 信用返済買いの約定株数 |

### Response Format
```json
{
  "breakdown": [
    {
      "Date": "2015-04-01",
      "Code": "13010",
      "LongSellValue": 115164000.0,
      "ShortSellWithoutMarginValue": 93561000.0,
      "MarginSellNewValue": 6412000.0,
      "MarginSellCloseValue": 23009000.0,
      "LongBuyValue": 185114000.0,
      "MarginBuyNewValue": 35568000.0,
      "MarginBuyCloseValue": 17464000.0,
      "LongSellVolume": 415000.0,
      "ShortSellWithoutMarginVolume": 337000.0,
      "MarginSellNewVolume": 23000.0,
      "MarginSellCloseVolume": 83000.0,
      "LongBuyVolume": 667000.0,
      "MarginBuyNewVolume": 128000.0,
      "MarginBuyCloseVolume": 63000.0
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 3. Daily Margin Interest (日々公表信用取引残高)

**Endpoint**: `GET /v1/markets/daily_margin_interest`

**Description**: 各銘柄の信用取引残高（株数）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| PublishedDate | String | 公表日 |
| Code | String | 銘柄コード |
| ApplicationDate | String | 申込日 (YYYY-MM-DD, 信用取引残高基準時点) |
| PublishReason | Map | 公表の理由 (Restricted, DailyPublication, Monitoring, RestrictedByJSF, PrecautionByJSF, UnclearOrSecOnAlert) |
| ShortMarginOutstanding | Number | 売合計信用残高 |
| DailyChangeShortMarginOutstanding | Number/String | 前日比 売合計信用残高 (単位：株、前日公表なしの場合は"-") |
| ShortMarginOutstandingListedShareRatio | Number/String | 上場比 売合計信用残高 (単位：％、ETFの場合は"*") |
| LongMarginOutstanding | Number | 買合計信用残高 |
| DailyChangeLongMarginOutstanding | Number/String | 前日比 買合計信用残高 (単位：株、前日公表なしの場合は"-") |
| LongMarginOutstandingListedShareRatio | Number/String | 上場比 買合計信用残高 (単位：％、ETFの場合は"*") |
| ShortLongRatio | Number | 取組比率 (単位：％) |
| ShortNegotiableMarginOutstanding | Number | 一般信用取引売残高 |
| DailyChangeShortNegotiableMarginOutstanding | Number/String | 前日比 一般信用取引売残高 (単位：株、前日公表なしの場合は"-") |
| ShortStandardizedMarginOutstanding | Number | 制度信用取引売残高 |
| DailyChangeShortStandardizedMarginOutstanding | Number/String | 前日比 制度信用取引売残高 (単位：株、前日公表なしの場合は"-") |
| LongNegotiableMarginOutstanding | Number | 一般信用取引買残高 |
| DailyChangeLongNegotiableMarginOutstanding | Number/String | 前日比 一般信用取引買残高 (単位：株、前日公表なしの場合は"-") |
| LongStandardizedMarginOutstanding | Number | 制度信用取引買残高 |
| DailyChangeLongStandardizedMarginOutstanding | Number/String | 前日比 制度信用取引買残高 (単位：株、前日公表なしの場合は"-") |
| TSEMarginBorrowingAndLendingRegulationClassification | String | 東証信用貸借規制区分 |

### Response Format
```json
{
  "daily_margin_interest": [
    {
      "PublishedDate": "2024-02-08",
      "Code": "13260",
      "ApplicationDate": "2024-02-07",
      "PublishReason": {
        "Restricted": "0",
        "DailyPublication": "0",
        "Monitoring": "0",
        "RestrictedByJSF": "0",
        "PrecautionByJSF": "1",
        "UnclearOrSecOnAlert": "0"
      },
      "ShortMarginOutstanding": 11.0,
      "DailyChangeShortMarginOutstanding": 0.0,
      "ShortMarginOutstandingListedShareRatio": "*",
      "LongMarginOutstanding": 676.0,
      "DailyChangeLongMarginOutstanding": -20.0,
      "LongMarginOutstandingListedShareRatio": "*",
      "ShortLongRatio": 1.6,
      "ShortNegotiableMarginOutstanding": 0.0,
      "DailyChangeShortNegotiableMarginOutstanding": 0.0,
      "ShortStandardizedMarginOutstanding": 11.0,
      "DailyChangeShortStandardizedMarginOutstanding": 0.0,
      "LongNegotiableMarginOutstanding": 192.0,
      "DailyChangeLongNegotiableMarginOutstanding": -20.0,
      "LongStandardizedMarginOutstanding": 484.0,
      "DailyChangeLongStandardizedMarginOutstanding": 0.0,
      "TSEMarginBorrowingAndLendingRegulationClassification": "001"
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 4. Daily Quotes (株価四本値)

**Endpoint**: `GET /v1/prices/daily_quotes`

**Description**: 株価データ（調整済み株価と調整前株価）。前場/後場別データはPremiumプランのみ。

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| Code | String | 銘柄コード |
| Open | Number | 始値（調整前） |
| High | Number | 高値（調整前） |
| Low | Number | 安値（調整前） |
| Close | Number | 終値（調整前） |
| UpperLimit | String | 日通ストップ高フラグ (0: ストップ高以外, 1: ストップ高) |
| LowerLimit | String | 日通ストップ安フラグ (0: ストップ安以外, 1: ストップ安) |
| Volume | Number | 取引高（調整前） |
| TurnoverValue | Number | 取引代金 |
| AdjustmentFactor | Number | 調整係数 (株式分割1:2の場合、権利落ち日に"0.5") |
| AdjustmentOpen | Number | 調整済み始値 |
| AdjustmentHigh | Number | 調整済み高値 |
| AdjustmentLow | Number | 調整済み安値 |
| AdjustmentClose | Number | 調整済み終値 |
| AdjustmentVolume | Number | 調整済み取引高 |
| MorningOpen | Number | 前場始値 (Premiumプランのみ) |
| MorningHigh | Number | 前場高値 (Premiumプランのみ) |
| MorningLow | Number | 前場安値 (Premiumプランのみ) |
| MorningClose | Number | 前場終値 (Premiumプランのみ) |
| MorningUpperLimit | String | 前場ストップ高フラグ (Premiumプランのみ) |
| MorningLowerLimit | String | 前場ストップ安フラグ (Premiumプランのみ) |
| MorningVolume | Number | 前場売買高 (Premiumプランのみ) |
| MorningTurnoverValue | Number | 前場取引代金 (Premiumプランのみ) |
| MorningAdjustmentOpen | Number | 調整済み前場始値 (Premiumプランのみ) |
| MorningAdjustmentHigh | Number | 調整済み前場高値 (Premiumプランのみ) |
| MorningAdjustmentLow | Number | 調整済み前場安値 (Premiumプランのみ) |
| MorningAdjustmentClose | Number | 調整済み前場終値 (Premiumプランのみ) |
| MorningAdjustmentVolume | Number | 調整済み前場売買高 (Premiumプランのみ) |
| AfternoonOpen | Number | 後場始値 (Premiumプランのみ) |
| AfternoonHigh | Number | 後場高値 (Premiumプランのみ) |
| AfternoonLow | Number | 後場安値 (Premiumプランのみ) |
| AfternoonClose | Number | 後場終値 (Premiumプランのみ) |
| AfternoonUpperLimit | String | 後場ストップ高フラグ (Premiumプランのみ) |
| AfternoonLowerLimit | String | 後場ストップ安フラグ (Premiumプランのみ) |
| AfternoonVolume | Number | 後場売買高 (Premiumプランのみ) |
| AfternoonTurnoverValue | Number | 後場取引代金 (Premiumプランのみ) |
| AfternoonAdjustmentOpen | Number | 調整済み後場始値 (Premiumプランのみ) |
| AfternoonAdjustmentHigh | Number | 調整済み後場高値 (Premiumプランのみ) |
| AfternoonAdjustmentLow | Number | 調整済み後場安値 (Premiumプランのみ) |
| AfternoonAdjustmentClose | Number | 調整済み後場終値 (Premiumプランのみ) |
| AfternoonAdjustmentVolume | Number | 調整済み後場売買高 (Premiumプランのみ) |

### Response Format
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
      "MorningClose": 2045.5
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 5. Dividend (配当金情報)

**Endpoint**: `GET /v1/fins/dividend`

**Description**: 上場会社の配当（決定・予想）に関する情報

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| AnnouncementDate | String | 通知日時（年月日） (YYYY-MM-DD) |
| AnnouncementTime | String | 通知日時（時分） (HH:MI) |
| Code | String | 銘柄コード |
| ReferenceNumber | String | リファレンスナンバー（配当通知を一意に特定） |
| StatusCode | String | 更新区分コード (1: 新規, 2: 訂正, 3: 削除) |
| BoardMeetingDate | String | 取締役会決議日 |
| InterimFinalCode | String | 配当種類コード (1: 中間配当, 2: 期末配当) |
| ForecastResultCode | String | 予想／決定コード (1: 決定, 2: 予想) |
| InterimFinalTerm | String | 配当基準日年月 |
| GrossDividendRate | Number/String | １株当たり配当金額 (未定: "-", 非設定: 空文字) |
| RecordDate | String | 基準日 |
| ExDate | String | 権利落日 |
| ActualRecordDate | String | 権利確定日 |
| PayableDate | String | 支払開始予定日 (未定: "-", 非設定: 空文字) |
| CAReferenceNumber | String | ＣＡリファレンスナンバー（訂正・削除対象のリファレンスナンバー） |
| DistributionAmount | Number/String | 1株当たりの交付金銭等の額 (2014/2/24以降提供) |
| RetainedEarnings | Number/String | 1株当たりの利益剰余金の額 (2014/2/24以降提供) |
| DeemedDividend | Number/String | 1株当たりのみなし配当の額 (2014/2/24以降提供) |
| DeemedCapitalGains | Number/String | 1株当たりのみなし譲渡収入の額 (2014/2/24以降提供) |
| NetAssetDecreaseRatio | Number/String | 純資産減少割合 (2014/2/24以降提供) |
| CommemorativeSpecialCode | String | 記念配当/特別配当コード (1: 記念配当, 2: 特別配当, 3: 記念・特別配当, 0: 通常の配当) |
| CommemorativeDividendRate | Number/String | １株当たり記念配当金額 (2022/6/6以降提供) |
| SpecialDividendRate | Number/String | １株当たり特別配当金額 (2022/6/6以降提供) |

### Response Format
```json
{
  "dividend": [
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
      "PayableDate": "-"
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 6. Futures (先物四本値)

**Endpoint**: `GET /v1/derivatives/futures`

**Description**: 先物に関する四本値、清算値段、理論価格情報（2016年7月19日以降の項目含む）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Code | String | 銘柄コード |
| DerivativeProductCategory | String | 先物商品区分 |
| Date | String | 取引日 (YYYY-MM-DD) |
| WholeDayOpen | Number | 日通し始値 |
| WholeDayHigh | Number | 日通し高値 |
| WholeDayLow | Number | 日通し安値 |
| WholeDayClose | Number | 日通し終値 |
| MorningSessionOpen | Number/String | 前場始値 (前後場取引対象でない場合は空文字) |
| MorningSessionHigh | Number/String | 前場高値 (前後場取引対象でない場合は空文字) |
| MorningSessionLow | Number/String | 前場安値 (前後場取引対象でない場合は空文字) |
| MorningSessionClose | Number/String | 前場終値 (前後場取引対象でない場合は空文字) |
| NightSessionOpen | Number/String | ナイト・セッション始値 (取引開始日初日は空文字) |
| NightSessionHigh | Number/String | ナイト・セッション高値 (取引開始日初日は空文字) |
| NightSessionLow | Number/String | ナイト・セッション安値 (取引開始日初日は空文字) |
| NightSessionClose | Number/String | ナイト・セッション終値 (取引開始日初日は空文字) |
| DaySessionOpen | Number | 日中始値 |
| DaySessionHigh | Number | 日中高値 |
| DaySessionLow | Number | 日中安値 |
| DaySessionClose | Number | 日中終値 |
| Volume | Number | 取引高 |
| OpenInterest | Number | 建玉 |
| TurnoverValue | Number | 取引代金 |
| ContractMonth | String | 限月 (YYYY-MM) |
| Volume(OnlyAuction) | Number | 立会内取引高 (2016/7/19以降) |
| EmergencyMarginTriggerDivision | String | 緊急取引証拠金発動区分 (001: 緊急時, 002: 清算価格算出時) |
| LastTradingDay | String | 取引最終年月日 (YYYY-MM-DD, 2016/7/19以降) |
| SpecialQuotationDay | String | SQ日 (YYYY-MM-DD, 2016/7/19以降) |
| SettlementPrice | Number | 清算値段 (2016/7/19以降) |
| CentralContractMonthFlag | String | 中心限月フラグ (1:中心限月, 0:その他, 2016/7/19以降) |

### Response Format
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
      "Volume": 42910.0,
      "OpenInterest": 479812.0,
      "TurnoverValue": 1217918971856.0,
      "ContractMonth": "2024-09",
      "SettlementPrice": 2829.0,
      "CentralContractMonthFlag": "1"
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 7. Index Option (日経225オプション)

**Endpoint**: `GET /v1/option/index_option`

**Description**: 日経225オプションの四本値、清算値段、理論価格（Weeklyオプション・フレックスオプションを除く）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 取引日 (YYYY-MM-DD) |
| Code | String | 銘柄コード |
| WholeDayOpen | Number | 日通し始値 |
| WholeDayHigh | Number | 日通し高値 |
| WholeDayLow | Number | 日通し安値 |
| WholeDayClose | Number | 日通し終値 |
| NightSessionOpen | Number/String | ナイト・セッション始値 (取引開始日初日は空文字) |
| NightSessionHigh | Number/String | ナイト・セッション高値 (取引開始日初日は空文字) |
| NightSessionLow | Number/String | ナイト・セッション安値 (取引開始日初日は空文字) |
| NightSessionClose | Number/String | ナイト・セッション終値 (取引開始日初日は空文字) |
| DaySessionOpen | Number | 日中始値 |
| DaySessionHigh | Number | 日中高値 |
| DaySessionLow | Number | 日中安値 |
| DaySessionClose | Number | 日中終値 |
| Volume | Number | 取引高 |
| OpenInterest | Number | 建玉 |
| TurnoverValue | Number | 取引代金 |
| ContractMonth | String | 限月 (YYYY-MM) |
| StrikePrice | Number | 権利行使価格 |
| Volume(OnlyAuction) | Number | 立会内取引高 (2016/7/19以降) |
| EmergencyMarginTriggerDivision | String | 緊急取引証拠金発動区分 (001: 緊急時, 002: 清算価格算出時) |
| PutCallDivision | String | プットコール区分 (1: プット, 2: コール) |
| LastTradingDay | String | 取引最終年月日 (YYYY-MM-DD, 2016/7/19以降) |
| SpecialQuotationDay | String | SQ日 (YYYY-MM-DD, 2016/7/19以降) |
| SettlementPrice | Number | 清算値段 (2016/7/19以降) |
| TheoreticalPrice | Number | 理論価格 (2016/7/19以降) |
| BaseVolatility | Number | 基準ボラティリティ (2016/7/19以降) |
| UnderlyingPrice | Number | 原証券価格 (2016/7/19以降) |
| ImpliedVolatility | Number | インプライドボラティリティ (2016/7/19以降) |
| InterestRate | Number | 理論価格計算用金利 (2016/7/19以降) |

### Response Format
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
      "Volume": 0.0,
      "OpenInterest": 330.0,
      "ContractMonth": "2025-06",
      "StrikePrice": 20000.0,
      "PutCallDivision": "1",
      "SettlementPrice": 980.0,
      "TheoreticalPrice": 974.641,
      "BaseVolatility": 17.93025
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 8. Indices (指数四本値)

**Endpoint**: `GET /v1/indices`

**Description**: 各種指数の四本値データ

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| Code | String | 指数コード |
| Open | Number | 始値 |
| High | Number | 高値 |
| Low | Number | 安値 |
| Close | Number | 終値 |

### Response Format
```json
{
  "indices": [
    {
      "Date": "2023-12-01",
      "Code": "0028",
      "Open": 1199.18,
      "High": 1202.58,
      "Low": 1195.01,
      "Close": 1200.17
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 9. Listed Info (上場銘柄一覧)

**Endpoint**: `GET /v1/listed/info`

**Description**: 過去時点・当日・翌営業日時点の銘柄情報（翌営業日は17:30以降取得可能）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 情報適用年月日 (YYYY-MM-DD) |
| Code | String | 銘柄コード |
| CompanyName | String | 会社名 |
| CompanyNameEnglish | String | 会社名（英語） |
| Sector17Code | String | 17業種コード |
| Sector17CodeName | String | 17業種コード名 |
| Sector33Code | String | 33業種コード |
| Sector33CodeName | String | 33業種コード名 |
| ScaleCategory | String | 規模コード |
| MarketCode | String | 市場区分コード |
| MarketCodeName | String | 市場区分名 |
| MarginCode | String | 貸借信用区分 (1: 信用, 2: 貸借, 3: その他, Standard/Premiumプランのみ) |
| MarginCodeName | String | 貸借信用区分名 (Standard/Premiumプランのみ) |

### Response Format
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
      "MarginCodeName": "信用"
    }
  ]
}
```

---

## 10. Options (オプション四本値)

**Endpoint**: `GET /v1/derivatives/options`

**Description**: オプションの四本値、清算値段、理論価格情報

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Code | String | 銘柄コード |
| DerivativeProductCategory | String | オプション商品区分 |
| UnderlyingSSO | String | 有価証券オプション対象銘柄 (有価証券オプション以外は"-") |
| Date | String | 取引日 (YYYY-MM-DD) |
| WholeDayOpen | Number | 日通し始値 |
| WholeDayHigh | Number | 日通し高値 |
| WholeDayLow | Number | 日通し安値 |
| WholeDayClose | Number | 日通し終値 |
| MorningSessionOpen | Number/String | 前場始値 (前後場取引対象でない場合は空文字) |
| MorningSessionHigh | Number/String | 前場高値 (前後場取引対象でない場合は空文字) |
| MorningSessionLow | Number/String | 前場安値 (前後場取引対象でない場合は空文字) |
| MorningSessionClose | Number/String | 前場終値 (前後場取引対象でない場合は空文字) |
| NightSessionOpen | Number/String | ナイト・セッション始値 (取引開始日初日は空文字) |
| NightSessionHigh | Number/String | ナイト・セッション高値 (取引開始日初日は空文字) |
| NightSessionLow | Number/String | ナイト・セッション安値 (取引開始日初日は空文字) |
| NightSessionClose | Number/String | ナイト・セッション終値 (取引開始日初日は空文字) |
| DaySessionOpen | Number | 日中始値 |
| DaySessionHigh | Number | 日中高値 |
| DaySessionLow | Number | 日中安値 |
| DaySessionClose | Number | 日中終値 |
| Volume | Number | 取引高 |
| OpenInterest | Number | 建玉 |
| TurnoverValue | Number | 取引代金 |
| ContractMonth | String | 限月 (YYYY-MM, 日経225miniの場合は週表記: 2024-51) |
| StrikePrice | Number | 権利行使価格 |
| Volume(OnlyAuction) | Number | 立会内取引高 (2016/7/19以降) |
| EmergencyMarginTriggerDivision | String | 緊急取引証拠金発動区分 (001: 緊急時, 002: 清算価格算出時) |
| PutCallDivision | String | プットコール区分 (1: プット, 2: コール) |
| LastTradingDay | String | 取引最終年月日 (YYYY-MM-DD, 2016/7/19以降) |
| SpecialQuotationDay | String | SQ日 (YYYY-MM-DD, 2016/7/19以降) |
| SettlementPrice | Number | 清算値段 (2016/7/19以降) |
| TheoreticalPrice | Number | 理論価格 (2016/7/19以降) |
| BaseVolatility | Number | 基準ボラティリティ (2016/7/19以降) |
| UnderlyingPrice | Number | 原証券価格 (2016/7/19以降) |
| ImpliedVolatility | Number | インプライドボラティリティ (2016/7/19以降) |
| InterestRate | Number | 理論価格計算用金利 (2016/7/19以降) |
| CentralContractMonthFlag | String | 中心限月フラグ (1:中心限月, 0:その他, 2016/7/19以降) |

### Response Format
```json
{
  "options": [
    {
      "Code": "140014505",
      "DerivativesProductCategory": "TOPIXE",
      "UnderlyingSSO": "-",
      "Date": "2024-07-23",
      "WholeDayOpen": 0.0,
      "Volume": 0.0,
      "OpenInterest": 0.0,
      "ContractMonth": "2025-01",
      "StrikePrice": 2450.0,
      "PutCallDivision": "2",
      "SettlementPrice": 377.0,
      "TheoreticalPrice": 380.3801
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 11. Prices AM (前場四本値)

**Endpoint**: `GET /v1/prices/prices_am`

**Description**: 前場終了時の株価データ（当日データは翌日6:00まで取得可能）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| Code | String | 銘柄コード |
| MorningOpen | Number | 前場始値 |
| MorningHigh | Number | 前場高値 |
| MorningLow | Number | 前場安値 |
| MorningClose | Number | 前場終値 |
| MorningVolume | Number | 前場売買高 |
| MorningTurnoverValue | Number | 前場取引代金 |

### Response Format
```json
{
  "prices_am": [
    {
      "Date": "2023-03-20",
      "Code": "39400",
      "MorningOpen": 232.0,
      "MorningHigh": 244.0,
      "MorningLow": 232.0,
      "MorningClose": 240.0,
      "MorningVolume": 52600.0,
      "MorningTurnoverValue": 12518800.0
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 12. Short Selling (業種別空売り比率)

**Endpoint**: `GET /v1/markets/short_selling`

**Description**: 日々の業種（セクター）別の空売り比率売買代金（円単位）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| Sector33Code | String | 33業種コード |
| SellingExcludingShortSellingTurnoverValue | Number | 実注文の売買代金 |
| ShortSellingWithRestrictionsTurnoverValue | Number | 価格規制有りの空売り売買代金 |
| ShortSellingWithoutRestrictionsTurnoverValue | Number | 価格規制無しの空売り売買代金 |

### Response Format
```json
{
  "short_selling": [
    {
      "Date": "2022-10-25",
      "Sector33Code": "0050",
      "SellingExcludingShortSellingTurnoverValue": 1333126400.0,
      "ShortSellingWithRestrictionsTurnoverValue": 787355200.0,
      "ShortSellingWithoutRestrictionsTurnoverValue": 149084300.0
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 13. Short Selling Positions (空売り残高報告)

**Endpoint**: `GET /v1/markets/short_selling_positions`

**Description**: 空売り残高報告（残高割合0.5%以上のみ）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| DisclosedDate | String | 日付（公表日） (YYYY-MM-DD) |
| CalculatedDate | String | 日付（計算日） (YYYY-MM-DD) |
| Code | String | 銘柄コード (5桁) |
| ShortSellerName | String | 商号・名称・氏名 |
| ShortSellerAddress | String | 住所・所在地 |
| DiscretionaryInvestmentContractorName | String | 委託者・投資一任契約の相手方の商号・名称・氏名 |
| DiscretionaryInvestmentContractorAddress | String | 委託者・投資一任契約の相手方の住所・所在地 |
| InvestmentFundName | String | 信託財産・運用財産の名称 |
| ShortPositionsToSharesOutstandingRatio | Number | 空売り残高割合 |
| ShortPositionsInSharesNumber | Number | 空売り残高数量 |
| ShortPositionsInTradingUnitsNumber | Number | 空売り残高売買単位数 |
| CalculationInPreviousReportingDate | String | 直近計算年月日 (YYYY-MM-DD) |
| ShortPositionsInPreviousReportingRatio | Number | 直近空売り残高割合 |
| Notes | String | 備考 |

### Response Format
```json
{
  "short_selling_positions": [
    {
      "DisclosedDate": "2024-08-01",
      "CalculatedDate": "2024-07-31",
      "Code": "13660",
      "ShortSellerName": "個人",
      "ShortSellerAddress": "",
      "ShortPositionsToSharesOutstandingRatio": 0.0053,
      "ShortPositionsInSharesNumber": 140000,
      "ShortPositionsInTradingUnitsNumber": 140000,
      "CalculationInPreviousReportingDate": "2024-07-22",
      "ShortPositionsInPreviousReportingRatio": 0.0043,
      "Notes": ""
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 14. Statements (財務情報)

**Endpoint**: `GET /v1/fins/statements`

**Description**: 上場企業の四半期毎の決算短信サマリーや業績・配当情報

**Note**: 項目名は日本基準（JGAAP）基準。IFRSや米国基準では経常利益等が空欄。

### Response Fields (Primary Fields)

| Field | Type | Description |
|-------|------|-------------|
| DisclosedDate | String | 開示日 |
| DisclosedTime | String | 開示時刻 |
| LocalCode | String | 銘柄コード（5桁） |
| DisclosureNumber | String | 開示番号（昇順にソート済み） |
| TypeOfDocument | String | 開示書類種別 |
| TypeOfCurrentPeriod | String | 当会計期間の種類 [1Q, 2Q, 3Q, 4Q, 5Q, FY] |
| CurrentPeriodStartDate | String | 当会計期間開始日 |
| CurrentPeriodEndDate | String | 当会計期間終了日 |
| CurrentFiscalYearStartDate | String | 当事業年度開始日 |
| CurrentFiscalYearEndDate | String | 当事業年度終了日 |
| NextFiscalYearStartDate | String | 翌事業年度開始日 |
| NextFiscalYearEndDate | String | 翌事業年度終了日 |

### Financial Metrics Fields

| Field | Type | Description |
|-------|------|-------------|
| NetSales | String | 売上高 |
| OperatingProfit | String | 営業利益 |
| OrdinaryProfit | String | 経常利益 |
| Profit | String | 当期純利益 |
| EarningsPerShare | String | 一株あたり当期純利益 |
| DilutedEarningsPerShare | String | 潜在株式調整後一株あたり当期純利益 |
| TotalAssets | String | 総資産 |
| Equity | String | 純資産 |
| EquityToAssetRatio | String | 自己資本比率 |
| BookValuePerShare | String | 一株あたり純資産 |
| CashFlowsFromOperatingActivities | String | 営業活動によるキャッシュ・フロー |
| CashFlowsFromInvestingActivities | String | 投資活動によるキャッシュ・フロー |
| CashFlowsFromFinancingActivities | String | 財務活動によるキャッシュ・フロー |
| CashAndEquivalents | String | 現金及び現金同等物期末残高 |

### Dividend Fields (18 fields)

Includes actual results (ResultDividendPerShare*), forecasts (ForecastDividendPerShare*), and next year forecasts (NextYearForecastDividendPerShare*) for quarters and annual periods.

### Forecast Fields (20 fields)

Includes current year forecasts (ForecastNetSales*, ForecastOperatingProfit*, etc.) and next year forecasts for 2Q and fiscal year end.

### Corporate Action Fields

| Field | Type | Description |
|-------|------|-------------|
| MaterialChangesInSubsidiaries | String | 期中における重要な子会社の異動 |
| SignificantChangesInTheScopeOfConsolidation | String | 期中における連結範囲の重要な変更 (2024/7/22以降) |
| ChangesBasedOnRevisionsOfAccountingStandard | String | 会計基準等の改正に伴う会計方針の変更 |
| ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard | String | 会計基準等の改正に伴う変更以外の会計方針の変更 |
| ChangesInAccountingEstimates | String | 会計上の見積りの変更 |
| RetrospectiveRestatement | String | 修正再表示 |

### Share Information Fields

| Field | Type | Description |
|-------|------|-------------|
| NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock | String | 期末発行済株式数 |
| NumberOfTreasuryStockAtTheEndOfFiscalYear | String | 期末自己株式数 |
| AverageNumberOfShares | String | 期中平均株式数 |

### Non-Consolidated Fields (35+ fields)

Non-consolidated versions of all major financial metrics (NonConsolidatedNetSales, NonConsolidatedOperatingProfit, etc.) with actual, forecast, and next year forecast variants.

### Response Format (Abbreviated)
```json
{
  "statements": [
    {
      "DisclosedDate": "2023-01-30",
      "DisclosedTime": "12:00:00",
      "LocalCode": "86970",
      "DisclosureNumber": "20230127594871",
      "TypeOfDocument": "3QFinancialStatements_Consolidated_IFRS",
      "TypeOfCurrentPeriod": "3Q",
      "NetSales": "100529000000",
      "OperatingProfit": "51765000000",
      "OrdinaryProfit": "",
      "Profit": "35175000000",
      "EarningsPerShare": "66.76",
      "TotalAssets": "79205861000000",
      "Equity": "320021000000"
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 15. TOPIX (TOPIX指数)

**Endpoint**: `GET /v1/indices/topix`

**Description**: TOPIXの日通しの四本値

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| Open | Number | 始値 |
| High | Number | 高値 |
| Low | Number | 安値 |
| Close | Number | 終値 |

### Response Format
```json
{
  "topix": [
    {
      "Date": "2022-06-28",
      "Open": 1885.52,
      "High": 1907.38,
      "Low": 1885.32,
      "Close": 1907.38
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 16. Trades Spec (投資部門別情報)

**Endpoint**: `GET /v1/markets/trades_spec`

**Description**: 投資部門別売買状況（株式・金額、単位：千円）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| PublishedDate | String | 公表日 (YYYY-MM-DD) |
| StartDate | String | 開始日 (YYYY-MM-DD) |
| EndDate | String | 終了日 (YYYY-MM-DD) |
| Section | String | 市場名 |
| ProprietarySales | Number | 自己計_売 |
| ProprietaryPurchases | Number | 自己計_買 |
| ProprietaryTotal | Number | 自己計_合計 |
| ProprietaryBalance | Number | 自己計_差引 |
| BrokerageSales | Number | 委託計_売 |
| BrokeragePurchases | Number | 委託計_買 |
| BrokerageTotal | Number | 委託計_合計 |
| BrokerageBalance | Number | 委託計_差引 |
| TotalSales | Number | 総計_売 |
| TotalPurchases | Number | 総計_買 |
| TotalTotal | Number | 総計_合計 |
| TotalBalance | Number | 総計_差引 |
| IndividualsSales | Number | 個人_売 |
| IndividualsPurchases | Number | 個人_買 |
| IndividualsTotal | Number | 個人_合計 |
| IndividualsBalance | Number | 個人_差引 |
| ForeignersSales | Number | 海外投資家_売 |
| ForeignersPurchases | Number | 海外投資家_買 |
| ForeignersTotal | Number | 海外投資家_合計 |
| ForeignersBalance | Number | 海外投資家_差引 |
| SecuritiesCosSales | Number | 証券会社_売 |
| SecuritiesCosPurchases | Number | 証券会社_買 |
| SecuritiesCosTotal | Number | 証券会社_合計 |
| SecuritiesCosBalance | Number | 証券会社_差引 |
| InvestmentTrustsSales | Number | 投資信託_売 |
| InvestmentTrustsPurchases | Number | 投資信託_買 |
| InvestmentTrustsTotal | Number | 投資信託_合計 |
| InvestmentTrustsBalance | Number | 投資信託_差引 |
| BusinessCosSales | Number | 事業法人_売 |
| BusinessCosPurchases | Number | 事業法人_買 |
| BusinessCosTotal | Number | 事業法人_合計 |
| BusinessCosBalance | Number | 事業法人_差引 |
| OtherCosSales | Number | その他法人_売 |
| OtherCosPurchases | Number | その他法人_買 |
| OtherCosTotal | Number | その他法人_合計 |
| OtherCosBalance | Number | その他法人_差引 |
| InsuranceCosSales | Number | 生保・損保_売 |
| InsuranceCosPurchases | Number | 生保・損保_買 |
| InsuranceCosTotal | Number | 生保・損保_合計 |
| InsuranceCosBalance | Number | 生保・損保_差引 |
| CityBKsRegionalBKsEtcSales | Number | 都銀・地銀等_売 |
| CityBKsRegionalBKsEtcPurchases | Number | 都銀・地銀等_買 |
| CityBKsRegionalBKsEtcTotal | Number | 都銀・地銀等_合計 |
| CityBKsRegionalBKsEtcBalance | Number | 都銀・地銀等_差引 |
| TrustBanksSales | Number | 信託銀行_売 |
| TrustBanksPurchases | Number | 信託銀行_買 |
| TrustBanksTotal | Number | 信託銀行_合計 |
| TrustBanksBalance | Number | 信託銀行_差引 |
| OtherFinancialInstitutionsSales | Number | その他金融機関_売 |
| OtherFinancialInstitutionsPurchases | Number | その他金融機関_買 |
| OtherFinancialInstitutionsTotal | Number | その他金融機関_合計 |
| OtherFinancialInstitutionsBalance | Number | その他金融機関_差引 |

### Response Format (Abbreviated)
```json
{
  "trades_spec": [
    {
      "PublishedDate": "2017-01-13",
      "StartDate": "2017-01-04",
      "EndDate": "2017-01-06",
      "Section": "TSE1st",
      "ProprietarySales": 1311271004,
      "ProprietaryPurchases": 1453326508,
      "TotalSales": 8476800009,
      "TotalPurchases": 8483346362,
      "IndividualsSales": 1401711615,
      "ForeignersSales": 5094891735
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## 17. Trading Calendar (取引カレンダー)

**Endpoint**: `GET /v1/markets/trading_calendar`

**Description**: 東証およびOSEの営業日、休業日、祝日取引の有無

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 日付 (YYYY-MM-DD) |
| HolidayDivision | String | 休日区分 |

### Response Format
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

---

## 18. Weekly Margin Interest (信用取引週末残高)

**Endpoint**: `GET /v1/markets/weekly_margin_interest`

**Description**: 週末時点での各銘柄の信用取引残高（株数）

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | String | 申込日付 (YYYY-MM-DD, 通常は金曜日付) |
| Code | String | 銘柄コード |
| ShortMarginTradeVolume | Number | 売合計信用取引週末残高 |
| LongMarginTradeVolume | Number | 買合計信用取引週末残高 |
| ShortNegotiableMarginTradeVolume | Number | 売一般信用取引週末残高 |
| LongNegotiableMarginTradeVolume | Number | 買一般信用取引週末残高 |
| ShortStandardizedMarginTradeVolume | Number | 売制度信用取引週末残高 |
| LongStandardizedMarginTradeVolume | Number | 買制度信用取引週末残高 |
| IssueType | String | 銘柄区分 (1: 信用銘柄, 2: 貸借銘柄, 3: その他) |

### Response Format
```json
{
  "weekly_margin_interest": [
    {
      "Date": "2023-02-17",
      "Code": "13010",
      "ShortMarginTradeVolume": 4100.0,
      "LongMarginTradeVolume": 27600.0,
      "ShortNegotiableMarginTradeVolume": 1300.0,
      "LongNegotiableMarginTradeVolume": 7600.0,
      "ShortStandardizedMarginTradeVolume": 2800.0,
      "LongStandardizedMarginTradeVolume": 20000.0,
      "IssueType": "2"
    }
  ],
  "pagination_key": "value1.value2."
}
```

---

## Common Response Elements

### Pagination

All APIs support pagination via the `pagination_key` parameter:

```json
{
  "data": [...],
  "pagination_key": "value1.value2."
}
```

When `pagination_key` is present in the response, use it in the next request to retrieve additional data. Pagination continues until no `pagination_key` is returned.

### Error Responses

All APIs return standard error responses:

**400 Bad Request**:
```json
{
  "message": "<Error Message>"
}
```

**401 Unauthorized**:
```json
{
  "message": "The incoming token is invalid or expired."
}
```

**403 Forbidden**:
```json
{
  "message": "<Error Message>"
}
```

**413 Payload Too Large**:
```json
{
  "message": "Response data is too large. Specify parameters to reduce the acquired data range."
}
```

**500 Internal Server Error**:
```json
{
  "message": "Unexpected error. Please try again later."
}
```

---

## Notes

1. **Date Format**: All date fields use `YYYY-MM-DD` format
2. **Time Format**: Time fields use `HH:MI` or `HH:MI:SS` format
3. **Premium Plan Features**: Some fields are marked as Premium-only (e.g., morning/afternoon session data in daily_quotes)
4. **Null Values**: Fields may contain `null`, empty strings `""`, or special markers like `"-"` or `"*"` depending on data availability
5. **Data Availability**:
   - 2016/7/19 onwards: Futures/Options additional fields (settlement price, theoretical price, etc.)
   - 2020/10/1: Trading halted due to system issues - some data missing
   - 2022/4/4: Market segment revision - historical data adjusted
6. **Rate Limiting**: APIs have rate limits that adjust based on usage
7. **Response Compression**: Responses are Gzip-compressed by default

---

## References

- **Official Documentation**: https://jpx.gitbook.io/j-quants-ja/
- **JPX Market Data**: https://www.jpx.co.jp/
- **J-Quants API Portal**: https://api.jquants.com/

---

**End of Document**
