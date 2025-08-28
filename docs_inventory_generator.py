#!/usr/bin/env python3
"""
ドキュメント棚卸しツール - Gogooku3リポジトリの全.mdファイルを解析
"""

import os
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime
import subprocess
import argparse


def get_file_hash(filepath):
    """ファイルのSHA256ハッシュを取得"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return "error"


def extract_title(content):
    """Markdown内容からタイトル(h1)を抽出"""
    lines = content.split('\n')
    for line in lines[:20]:  # 最初の20行から探索
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
    return "no_title"


def detect_language(content, filepath):
    """言語を判定（日本語/英語/その他）"""
    # ファイル名パターンでの判定
    if '/ja/' in str(filepath) or '_ja.' in str(filepath):
        return 'ja'
    if '/en/' in str(filepath) or '_en.' in str(filepath):
        return 'en'
    
    # 内容での判定（簡易）
    japanese_chars = len(re.findall(r'[ひらがなカタカナ漢字]', content))
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', content))
    
    if japanese_chars > english_words * 0.1:
        return 'ja'
    elif english_words > 50:
        return 'en'
    else:
        return 'other'


def classify_doc_type(filepath, content):
    """ドキュメントタイプを分類"""
    path_str = str(filepath).lower()
    title = extract_title(content).lower()
    
    # パスベース判定
    if 'readme' in path_str:
        return 'guide'
    if 'specification' in path_str or 'spec' in path_str:
        return 'spec'
    if 'migration' in path_str or 'changelog' in path_str:
        return 'changelog'
    if 'adr' in path_str or 'rfc' in path_str:
        return 'adr'
    if 'api' in path_str:
        return 'api'
    if 'runbook' in path_str or 'operation' in path_str:
        return 'runbook'
    
    # 内容ベース判定
    if any(word in title for word in ['guide', 'tutorial', 'getting started']):
        return 'guide'
    if any(word in title for word in ['specification', 'design', 'architecture']):
        return 'spec'
    if any(word in title for word in ['api', 'reference']):
        return 'api'
    if any(word in title for word in ['decision', 'adr']):
        return 'adr'
    if any(word in title for word in ['report', 'status', 'plan']):
        return 'memo'
    
    return 'memo'


def count_links(content):
    """バックリンク・アウトバウンドリンクをカウント（簡易）"""
    # Markdownリンク [text](url) をカウント
    outbound = len(re.findall(r'\[.*?\]\(.*?\)', content))
    
    # バックリンクは別途解析が必要（ここでは0）
    backlinks = 0
    
    return backlinks, outbound


def get_last_commit_date(filepath):
    """最終コミット日を取得"""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci', str(filepath)],
            capture_output=True, text=True, cwd=filepath.parent
        )
        if result.returncode == 0:
            return result.stdout.strip().split()[0]  # YYYY-MM-DD部分
    except:
        pass
    return "unknown"


def analyze_markdown_files(repo_root):
    """リポジトリ内の全.mdファイルを解析"""
    repo_path = Path(repo_root)
    md_files = list(repo_path.glob('**/*.md'))
    
    results = []
    
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = ""
        
        # 統計情報収集
        size = filepath.stat().st_size if filepath.exists() else 0
        hash_value = get_file_hash(filepath)
        title = extract_title(content)
        language = detect_language(content, filepath)
        doc_type = classify_doc_type(filepath, content)
        last_commit = get_last_commit_date(filepath)
        backlinks, outbound_links = count_links(content)
        
        # 相対パス取得
        try:
            rel_path = filepath.relative_to(repo_path)
        except:
            rel_path = filepath
        
        results.append({
            'path': str(rel_path),
            'size': size,
            'sha256': hash_value[:16],  # 短縮版
            'title': title,
            'lang': language,
            'type': doc_type,
            'last_commit_date': last_commit,
            'backlinks_count': backlinks,
            'outbound_links_count': outbound_links
        })
    
    return results


def save_to_csv(results, output_path):
    """結果をCSVに保存"""
    fieldnames = [
        'path', 'size', 'sha256', 'title', 'lang', 'type',
        'last_commit_date', 'backlinks_count', 'outbound_links_count'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"📊 解析結果を {output_path} に保存しました")
    print(f"📁 合計 {len(results)} 個のMarkdownファイルを処理")


def main():
    parser = argparse.ArgumentParser(description='Markdown文書棚卸しツール')
    parser.add_argument('--repo', default='.', help='リポジトリルートパス')
    parser.add_argument('--out', default='docs_inventory.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    print(f"🔍 {args.repo} 配下のMarkdownファイルを解析中...")
    results = analyze_markdown_files(args.repo)
    save_to_csv(results, args.out)
    
    # サマリー表示
    print(f"\n📋 サマリー:")
    print(f"   総ファイル数: {len(results)}")
    print(f"   言語分布: {dict(sorted([(lang, sum(1 for r in results if r['lang'] == lang)) for lang in set(r['lang'] for r in results)]))}")
    print(f"   タイプ分布: {dict(sorted([(doc_type, sum(1 for r in results if r['type'] == doc_type)) for doc_type in set(r['type'] for r in results)]))}")


if __name__ == '__main__':
    main()