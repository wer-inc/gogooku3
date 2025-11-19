# API Bug Fixes - 2025-11-17

## Summary

Systematic audit of all JQuants API endpoints revealed **2 critical bugs** in `jquants_async_fetcher.py` causing data loss since 2025-11-04 (commit a7c2cfa).

## Bug 1: Financial Statements API (CRITICAL) ✅ FIXED

**Impact**: 100% data loss for share-related features (29 columns NULL)

### Root Cause: 4 Compounding Errors

1. **Wrong Endpoint**:
   - Used: `/fins/fs_details` (financial statement details - BS/PL/CF only)
   - Correct: `/fins/statements` (quarterly earnings summaries with share data)

2. **Wrong Response Key**:
   - Expected: `"fs_details"`
   - Correct: `"statements"` (as per API spec line 59)

3. **Wrong Code Field**:
   - Used: `item.get("Code")`
   - Correct: `item.get("LocalCode")` (5-digit code for statements API)

4. **Missing Share Column Extraction**:
   - Share columns exist at **top level** of response items, NOT nested in `FinancialStatement`
   - Columns: `NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock`, `NumberOfTreasuryStockAtTheEndOfFiscalYear`, `AverageNumberOfShares`

### Investigation Timeline

1. **Initial Discovery** (14:15 JST):
   - shares_master.parquet had 0 rows despite successful build
   - fs_details cache only had 9 columns (missing all share columns)

2. **User Direction** (14:18 JST):
   - User: "'/workspace/gogooku3/gogooku5/docs/external' 修正が必要ですね"
   - Checked local API docs: Found statements endpoint documentation

3. **Online Verification** (14:22 JST):
   - User: "https://jpx.gitbook.io/j-quants-ja/api-reference api仕様書を見てもらえますか"
   - Confirmed correct endpoint and response structure via WebFetch

4. **Historical Analysis** (14:28 JST):
   - User: "fs_detailsの問題はいつから起きていますか？"
   - Traced to commit a7c2cfa (2025-11-04) - bug existed since initial implementation (~13 days)

### Fixes Applied

**File**: `/workspace/gogooku3/gogooku5/data/src/builder/api/jquants_async_fetcher.py`

#### 1. Endpoint Fix (line 1534)

```python
# Before:
base_url = f"{self.base_url}/fins/fs_details"

# After:
base_url = f"{self.base_url}/fins/statements"
```

#### 2. Response Key Fix - Sequential Fetch (line 1675)

```python
# Before:
items = data.get("fs_details") or data.get("data") or []

# After:
items = data.get("statements") or data.get("fs_details") or data.get("data") or []
```

#### 3. Response Key Fix - Parallel Fetch (line 1478)

```python
# Before:
items = data.get("fs_details") or data.get("data") or []

# After:
items = data.get("statements") or data.get("fs_details") or data.get("data") or []
```

#### 4. Code Field Fix - Sequential Fetch (line 1678)

```python
# Before:
"Code": item.get("Code"),

# After:
"Code": item.get("LocalCode") or item.get("Code"),
```

#### 5. Code Field Fix - Parallel Fetch (line 1481)

```python
# Before:
"Code": item.get("Code"),

# After:
"Code": item.get("LocalCode") or item.get("Code"),
```

#### 6. Share Column Extraction - Sequential Fetch (lines 1687-1691)

```python
# Added after line 1686:
# Extract share columns from top level (not nested in FinancialStatement)
for candidate in issued_share_candidates + treasury_share_candidates + average_share_candidates:
    if candidate in item:
        flat[candidate] = item[candidate]
```

#### 7. Share Column Extraction - Parallel Fetch (lines 1490-1494)

```python
# Added after line 1489:
# Extract share columns from top level (not nested in FinancialStatement)
for candidate in share_candidates:
    if candidate in item:
        flat[candidate] = item[candidate]
```

#### 8. Parallel Fetch Function Signature (line 1437)

```python
# Before:
async def _get_fs_details_parallel(
    self,
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    date_list: list[str],
    extract_fn: Callable[[dict], dict],
) -> list[dict]:

# After:
async def _get_fs_details_parallel(
    self,
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    date_list: list[str],
    extract_fn: Callable[[dict], dict],
    share_candidates: tuple[str, ...] = (),
) -> list[dict]:
```

#### 9. Parallel Fetch Call (lines 1652-1655)

```python
# Before:
rows = await self._get_fs_details_parallel(
    session, base_url, headers, date_list, _extract_financials
)

# After:
all_share_candidates = issued_share_candidates + treasury_share_candidates + average_share_candidates
rows = await self._get_fs_details_parallel(
    session, base_url, headers, date_list, _extract_financials, all_share_candidates
)
```

### Verification

**Before Fix**:
```
fs_details cache: 9 columns (missing all share columns)
shares_master.parquet: 0 rows
```

**After Fix**:
```
shares_master.parquet: 7,444 rows, 3,976 unique codes
Date range: 2024-10-28 → 2025-04-01
shares_total: 0.0% NULL
shares_free_float: 0.5% NULL
```

## Bug 2: Dividend API (MEDIUM) ✅ FIXED

**Impact**: 100% data loss for dividend features

### Root Cause: Response Key Mismatch

- **Documentation**: `/fins/dividend` with response key `"dividend"` (singular)
- **Implementation**: Used `"dividends"` (plural) at lines 1373 and 1395
- **Evidence**: Online API spec (https://jpx.gitbook.io/j-quants-ja/api-reference/dividend) confirms singular form

### Fixes Applied

#### Line 1373 (range-based fetch)

```python
# Before:
if status == 200 and isinstance(data, dict):
    rows.extend(data.get("dividends") or data.get("data") or [])

# After:
if status == 200 and isinstance(data, dict):
    rows.extend(data.get("dividend") or data.get("dividends") or data.get("data") or [])
```

#### Line 1395 (daily pagination fallback)

```python
# Before:
if status != 200 or not isinstance(data, dict):
    break
items = data.get("dividends") or data.get("data") or []

# After:
if status != 200 or not isinstance(data, dict):
    break
items = data.get("dividend") or data.get("dividends") or data.get("data") or []
```

### Verification

**Before Fix**:
```
[2025-11-17 13:09:24] WARNING builder.api.data_sources - Dividend API returned zero rows for 2024-10-28 → 2025-03-31
```

**After Fix**: Build in progress (14:40+ JST) - will verify dividend data retrieval

## Complete API Endpoint Audit Results

| API | Endpoint | Expected Key | Implementation | Status |
|-----|----------|--------------|----------------|--------|
| **statements** | `/fins/statements` | `"statements"` | ~~`"fs_details"`~~ | ✅ **FIXED** |
| **dividend** | `/fins/dividend` | `"dividend"` | ~~`"dividends"`~~ | ✅ **FIXED** |
| **breakdown** | `/markets/breakdown` | `"breakdown"` | `"breakdown"` | ✅ Correct |
| **indices** | `/indices` | `"indices"` | `"indices"` | ✅ Correct |
| **prices_am** | `/prices/prices_am` | `"prices_am"` | `"prices_am"` | ✅ Correct |
| **index_option** | `/option/index_option` | `"index_option"` | `"index_option"` | ✅ Correct |

## Methodology

1. **Systematic Audit**: Checked all 6 active API endpoints in jquants_async_fetcher.py
2. **Documentation Cross-Reference**: Verified against both local docs (`/workspace/gogooku3/gogooku5/docs/external/jquants_api/`) and online API spec (https://jpx.gitbook.io/j-quants-ja/api-reference)
3. **Code Inspection**: Grep'd for response key extraction patterns: `data.get("xxx")`
4. **Historical Analysis**: Traced bug origin via git log

## User Guidance

**User Request** (14:32 JST):
> "fs_detailsの問題と同じ問題ほかにもありそうですね。調べてもらえますか"
> (There might be similar problems elsewhere. Can you investigate?)

**Response**: Conducted full audit, found and fixed 2 bugs (statements + dividend)

## Impact Assessment

### Before Fixes (2025-11-04 to 2025-11-17)

- **Duration**: 13 days
- **Affected Builds**: All dataset builds during this period
- **Data Loss**:
  - 29 share-related features: 100% NULL
  - Dividend features: 100% NULL
  - Total: ~35 columns affected

### After Fixes (2025-11-17+)

- ✅ shares_master.parquet: 7,444 rows (3,976 unique codes)
- ✅ Dividend data: Pending verification in ongoing build
- ✅ Schema hash updated: `0cd3124303aa255f` → `971c93f4459ef487`

## Recommendations

1. **Immediate**: Rebuild all datasets from 2025-11-04 onwards
2. **Short-term**: Add API response key validation tests
3. **Long-term**: Implement API response schema validation against official specs

## Related Files

- **Main Fix**: `/workspace/gogooku3/gogooku5/data/src/builder/api/jquants_async_fetcher.py`
- **Rebuilt Cache**: `/workspace/gogooku3/output_g5/cache/shares_master.parquet`
- **API Documentation**:
  - `/workspace/gogooku3/gogooku5/docs/external/jquants_api/j-quants-ja/api-reference/statements/index.md`
  - `/workspace/gogooku3/gogooku5/docs/external/jquants_api/j-quants-ja/api-reference/dividend/index.md`

## Commit Information

- **Bug Introduced**: a7c2cfa (2025-11-04) - Initial gogooku5 implementation
- **Bug Fixed**: (pending commit) - 2025-11-17
- **Files Modified**: 1 (`jquants_async_fetcher.py`)
- **Lines Changed**: 13 locations (9 for statements, 2 for dividend, 2 for function signatures)

## Testing

### Manual Verification

1. ✅ shares_master.parquet rebuild successful (7,444 rows)
2. 🔄 2025Q1 full dataset rebuild in progress (started 14:40 JST)
3. ⏳ Dividend data verification pending

### Expected Outcomes

- ✅ shares_total: 0.0% NULL (vs 100% before)
- ✅ shares_free_float: 0.5% NULL (vs 100% before)
- ⏳ dividend_yield: Expected >0% NULL (vs 100% before)

## Lessons Learned

1. **API Spec Compliance**: Always cross-reference implementation against official API documentation
2. **Response Structure Validation**: Different endpoints have different response structures (top-level vs nested)
3. **Field Name Conventions**: Some APIs use LocalCode (5-digit) vs Code (standard)
4. **Backward Compatibility**: Maintain fallbacks (`"dividend"` or `"dividends"`) for robustness
5. **Systematic Auditing**: User's intuition was correct - similar bugs existed in multiple endpoints

## Next Steps

1. ✅ Monitor ongoing 2025Q1 build completion (~15-25 min remaining)
2. ⏳ Quality validation after build completes
3. ⏳ Git commit with comprehensive message
4. 📋 Add API endpoint tests to prevent regression

---

**Generated**: 2025-11-17 14:40 JST
**Author**: Claude Code (autonomous investigation)
**User**: gogooku3 project maintainer
