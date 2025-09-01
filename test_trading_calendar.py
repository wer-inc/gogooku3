#!/usr/bin/env python3
"""
Trading Calendar APIのテスト
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import json

load_dotenv()

async def test_trading_calendar():
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
        
        # Trading Calendar取得テスト
        url = "https://api.jquants.com/v1/markets/trading_calendar"
        headers = {"Authorization": f"Bearer {id_token}"}
        
        # テスト1: 短期間（2024年12月）
        params = {
            "from": "20241202",
            "to": "20241206"
        }
        
        print(f"\nテスト1: {params}")
        async with session.get(url, headers=headers, params=params) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response keys: {data.keys()}")
            if "trading_calendar" in data:
                print(f"Calendar entries: {len(data['trading_calendar'])}")
                # 最初の5件を表示
                for entry in data['trading_calendar'][:5]:
                    print(f"  {entry}")
        
        # テスト2: holidaydivision指定あり
        params = {
            "holidaydivision": "0",  # 営業日のみ
            "from": "20241202",
            "to": "20241206"
        }
        
        print(f"\nテスト2: {params}")
        async with session.get(url, headers=headers, params=params) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            if "trading_calendar" in data:
                print(f"営業日のみ: {len(data['trading_calendar'])}件")
                for entry in data['trading_calendar']:
                    print(f"  {entry}")

if __name__ == "__main__":
    asyncio.run(test_trading_calendar())