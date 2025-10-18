"""
SectionMapperのテストケース
MarketCodeからSectionへの正しいマッピングを検証
"""

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.flow_joiner import normalize_section_name
from features.section_mapper import SectionMapper


class TestSectionMapper:
    """SectionMapperのテスト"""

    def test_market_code_mapping(self):
        """MarketCodeの基本的なマッピングをテスト"""
        mapper = SectionMapper()

        # 旧市場コードのテスト
        assert mapper.MARKET_TO_SECTION["0101"] == "TSE1st"  # 東証一部
        assert mapper.MARKET_TO_SECTION["0102"] == "TSE2nd"  # 東証二部
        assert mapper.MARKET_TO_SECTION["0104"] == "TSEMothers"  # マザーズ
        assert mapper.MARKET_TO_SECTION["0105"] == "TSEPro"  # TOKYO PRO
        assert (
            mapper.MARKET_TO_SECTION["0106"] == "JASDAQStandard"
        )  # JASDAQ スタンダード
        assert mapper.MARKET_TO_SECTION["0107"] == "JASDAQGrowth"  # JASDAQ グロース
        assert mapper.MARKET_TO_SECTION["0109"] == "Other"  # その他

        # 新市場コードのテスト
        assert mapper.MARKET_TO_SECTION["0111"] == "TSEPrime"  # プライム
        assert mapper.MARKET_TO_SECTION["0112"] == "TSEStandard"  # スタンダード
        assert mapper.MARKET_TO_SECTION["0113"] == "TSEGrowth"  # グロース

    def test_create_section_mapping_old_market(self):
        """旧市場コードでのセクションマッピング作成テスト"""
        mapper = SectionMapper()

        # 2022年4月以前のデータ
        listed_info_df = pl.DataFrame(
            {
                "Code": ["1301", "7203", "4661", "2121", "3765"],
                "MarketCode": [
                    "0101",
                    "0101",
                    "0104",
                    "0106",
                    "0107",
                ],  # 一部、一部、マザーズ、JASDAQ-S、JASDAQ-G
                "Date": [date(2022, 3, 31)] * 5,
            }
        )

        result = mapper.create_section_mapping(listed_info_df)

        # 期待されるマッピング
        expected_sections = {
            "1301": "TSE1st",
            "7203": "TSE1st",
            "4661": "TSEMothers",
            "2121": "JASDAQStandard",
            "3765": "JASDAQGrowth",
        }

        for code, expected_section in expected_sections.items():
            actual = result.filter(pl.col("Code") == code)["Section"][0]
            assert (
                actual == expected_section
            ), f"Code {code}: expected {expected_section}, got {actual}"

    def test_create_section_mapping_new_market(self):
        """新市場コードでのセクションマッピング作成テスト"""
        mapper = SectionMapper()

        # 2022年4月以降のデータ
        listed_info_df = pl.DataFrame(
            {
                "Code": ["1301", "7203", "4661", "2121"],
                "MarketCode": [
                    "0111",
                    "0111",
                    "0113",
                    "0112",
                ],  # プライム、プライム、グロース、スタンダード
                "Date": [date(2022, 4, 4)] * 4,
            }
        )

        result = mapper.create_section_mapping(listed_info_df)

        # 期待されるマッピング
        expected_sections = {
            "1301": "TSEPrime",
            "7203": "TSEPrime",
            "4661": "TSEGrowth",
            "2121": "TSEStandard",
        }

        for code, expected_section in expected_sections.items():
            actual = result.filter(pl.col("Code") == code)["Section"][0]
            assert (
                actual == expected_section
            ), f"Code {code}: expected {expected_section}, got {actual}"

    def test_market_transition_period(self):
        """市場再編期間のデータ処理テスト"""
        mapper = SectionMapper()

        # 市場再編をまたぐデータ
        listed_info_df = pl.DataFrame(
            {
                "Code": ["1301", "1301", "4661", "4661"],
                "MarketCode": [
                    "0101",
                    "0111",
                    "0104",
                    "0113",
                ],  # 一部→プライム、マザーズ→グロース
                "Date": [
                    date(2022, 3, 31),
                    date(2022, 4, 4),
                    date(2022, 3, 31),
                    date(2022, 4, 4),
                ],
            }
        )

        result = mapper.create_section_mapping(listed_info_df)

        # 1301: 東証一部 → プライム
        code_1301_old = result.filter(
            (pl.col("Code") == "1301") & (pl.col("valid_from") < date(2022, 4, 4))
        )
        if not code_1301_old.is_empty():
            assert code_1301_old["Section"][0] == "TSE1st"

        code_1301_new = result.filter(
            (pl.col("Code") == "1301") & (pl.col("valid_from") >= date(2022, 4, 4))
        )
        if not code_1301_new.is_empty():
            assert code_1301_new["Section"][0] == "TSEPrime"

        # 4661: マザーズ → グロース
        code_4661_old = result.filter(
            (pl.col("Code") == "4661") & (pl.col("valid_from") < date(2022, 4, 4))
        )
        if not code_4661_old.is_empty():
            assert code_4661_old["Section"][0] == "TSEMothers"

        code_4661_new = result.filter(
            (pl.col("Code") == "4661") & (pl.col("valid_from") >= date(2022, 4, 4))
        )
        if not code_4661_new.is_empty():
            assert code_4661_new["Section"][0] == "TSEGrowth"


class TestFlowJoinerNormalization:
    """flow_joinerのSection名正規化テスト"""

    def test_normalize_old_market_names(self):
        """旧市場名の正規化テスト"""
        # 東証一部のバリエーション
        assert normalize_section_name("TSE1st") == "TSE1st"
        assert normalize_section_name("TSE 1st") == "TSE1st"
        assert normalize_section_name("東証一部") == "TSE1st"
        assert normalize_section_name("市場一部") == "TSE1st"

        # 東証二部のバリエーション
        assert normalize_section_name("TSE2nd") == "TSE2nd"
        assert normalize_section_name("TSE 2nd") == "TSE2nd"
        assert normalize_section_name("東証二部") == "TSE2nd"
        assert normalize_section_name("市場二部") == "TSE2nd"

        # マザーズのバリエーション
        assert normalize_section_name("Mothers") == "TSEMothers"
        assert normalize_section_name("TSE Mothers") == "TSEMothers"
        assert normalize_section_name("TSEMothers") == "TSEMothers"
        assert normalize_section_name("マザーズ") == "TSEMothers"

        # JASDAQのバリエーション（結合脱落を避けるためStandardに寄せる）
        assert normalize_section_name("JASDAQ") == "JASDAQStandard"
        assert normalize_section_name("TSEJASDAQ") == "JASDAQStandard"
        assert normalize_section_name("JASDAQ Standard") == "JASDAQStandard"
        assert normalize_section_name("JASDAQスタンダード") == "JASDAQStandard"
        assert normalize_section_name("JASDAQ Growth") == "JASDAQGrowth"
        assert normalize_section_name("JASDAQグロース") == "JASDAQGrowth"

    def test_normalize_new_market_names(self):
        """新市場名の正規化テスト"""
        # プライムのバリエーション
        assert normalize_section_name("Prime") == "TSEPrime"
        assert normalize_section_name("Prime Market") == "TSEPrime"
        assert normalize_section_name("TSE Prime") == "TSEPrime"
        assert normalize_section_name("TSEPrime") == "TSEPrime"
        assert normalize_section_name("東証プライム") == "TSEPrime"
        assert normalize_section_name("プライム") == "TSEPrime"

        # スタンダードのバリエーション
        assert normalize_section_name("Standard") == "TSEStandard"
        assert normalize_section_name("Standard Market") == "TSEStandard"
        assert normalize_section_name("TSE Standard") == "TSEStandard"
        assert normalize_section_name("TSEStandard") == "TSEStandard"
        assert normalize_section_name("東証スタンダード") == "TSEStandard"
        assert normalize_section_name("スタンダード") == "TSEStandard"

        # グロースのバリエーション
        assert normalize_section_name("Growth") == "TSEGrowth"
        assert normalize_section_name("Growth Market") == "TSEGrowth"
        assert normalize_section_name("TSE Growth") == "TSEGrowth"
        assert normalize_section_name("TSEGrowth") == "TSEGrowth"
        assert normalize_section_name("東証グロース") == "TSEGrowth"
        assert normalize_section_name("グロース") == "TSEGrowth"

    def test_normalize_special_cases(self):
        """特殊ケースの正規化テスト"""
        # TokyoNagoya（東証および名証）- AllMarketに統合して結合脱落を回避
        assert normalize_section_name("TokyoNagoya") == "AllMarket"
        assert normalize_section_name("東証および名証") == "AllMarket"

        # 全市場
        assert normalize_section_name("All") == "AllMarket"
        assert normalize_section_name("ALL") == "AllMarket"
        assert normalize_section_name("All Market") == "AllMarket"
        assert normalize_section_name("ALL MARKET") == "AllMarket"
        assert normalize_section_name("Other") == "AllMarket"

        # None
        assert normalize_section_name(None) is None

        # 未知の値はそのまま返す
        assert normalize_section_name("UnknownMarket") == "UnknownMarket"

    def test_jasdaq_integration(self):
        """JASDAQ統合のテスト"""
        mapper = SectionMapper()

        # JASDAQStandard/Growth を TSEJASDAQ に統合
        assert mapper.get_section_for_trades_spec("JASDAQStandard") == "TSEJASDAQ"
        assert mapper.get_section_for_trades_spec("JASDAQGrowth") == "TSEJASDAQ"

        # その他はそのまま
        assert mapper.get_section_for_trades_spec("TSEPrime") == "TSEPrime"
        assert mapper.get_section_for_trades_spec("TSE1st") == "TSE1st"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
