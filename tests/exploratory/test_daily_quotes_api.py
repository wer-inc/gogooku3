#!/usr/bin/env python3
"""
Daily Quotes APIのパラメータ仕様確認
"""

import asyncio
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def test_daily_quotes():
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    async with aiohttp.ClientSession() as session:
        # 認証
        auth_url = "https://api.jquants.com/v1/token/auth_user"
        auth_payload = {"mailaddress": email, "password": password}

        async with session.post(auth_url, json=auth_payload) as response:
            data = await response.json()
            refresh_token = data["refreshToken"]

        # IDトークン取得
        refresh_url = "https://api.jquants.com/v1/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as response:
            data = await response.json()
            id_token = data["idToken"]

        print("✅ 認証成功")

        # Daily Quotes APIのテスト
        url = "https://api.jquants.com/v1/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {id_token}"}

        # テスト1: 日付のみ指定
        print("\nテスト1: 日付のみ指定")
        params = {"date": "20240108"}
        async with session.get(url, headers=headers, params=params) as response:
            print(f"  Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                quotes = data.get("daily_quotes", [])
                print(f"  取得レコード数: {len(quotes)}")
                if quotes:
                    print(f"  最初のレコード: {quotes[0]}")
            else:
                text = await response.text()
                print(f"  Error: {text}")

        # テスト2: 銘柄コード + 期間指定
        print("\nテスト2: 銘柄コード + 期間指定")
        params = {"code": "7203", "from": "20240104", "to": "20240110"}
        async with session.get(url, headers=headers, params=params) as response:
            print(f"  Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                quotes = data.get("daily_quotes", [])
                print(f"  取得レコード数: {len(quotes)}")
                for quote in quotes:
                    print(
                        f"    {quote.get('Date')}: {quote.get('Code')} - Close={quote.get('Close')}"
                    )

        # テスト3: 銘柄コードのみ（最新データ）
        print("\nテスト3: 銘柄コードのみ")
        params = {"code": "7203"}
        async with session.get(url, headers=headers, params=params) as response:
            print(f"  Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                quotes = data.get("daily_quotes", [])
                print(f"  取得レコード数: {len(quotes)}")
                if quotes:
                    print(f"  最新データ: {quotes[-1].get('Date')}")


if __name__ == "__main__":
    asyncio.run(test_daily_quotes())
