#!/usr/bin/env python3
"""
Markdownファイル重複検出ツール - exact/near/topic重複を検出
"""

import csv
import re
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import argparse
import json


def normalize_text(content):
    """テキストを正規化（見出し/コード/リンク除外）"""
    # コードブロック除去
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    # インラインコード除去  
    content = re.sub(r'`[^`]*`', '', content)
    # Markdownリンク除去
    content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
    # 見出し記号除去
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # 改行・空白正規化
    content = ' '.join(content.split())
    return content.lower()


def extract_keywords(content, title):
    """キーワードを抽出（タイトル・見出し・重要語）"""
    keywords = set()
    
    # タイトルから
    keywords.update(re.findall(r'\b[a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]{3,}\b', title.lower()))
    
    # 見出しから
    headings = re.findall(r'^#+\s*(.+)$', content, re.MULTILINE)
    for heading in headings:
        keywords.update(re.findall(r'\b[a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]{3,}\b', heading.lower()))
    
    # 重要そうなキーワード（技術用語）
    tech_words = re.findall(r'\b(gogooku|atft|gat|fan|jquants|dagster|mlflow|feast|clickhouse|minio|redis|postgresql|docker|make|training|pipeline|model)\b', content.lower())
    keywords.update(tech_words)
    
    return keywords


def detect_exact_duplicates(file_data):
    """完全重複検出（SHA256ベース）"""
    duplicates = []
    hash_groups = defaultdict(list)
    
    for item in file_data:
        hash_groups[item['sha256']].append(item)
    
    group_id = 1
    for hash_val, files in hash_groups.items():
        if len(files) > 1:
            # ファイルサイズで優先度決定（大きい方を優先）
            files.sort(key=lambda x: x['size'], reverse=True)
            preferred = files[0]['path']
            
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    duplicates.append({
                        'group_id': f'exact_{group_id}',
                        'type': 'exact-duplicate',
                        'path_a': files[i]['path'],
                        'path_b': files[j]['path'],
                        'similarity': 1.0,
                        'preferred': preferred
                    })
            group_id += 1
    
    return duplicates


def detect_near_duplicates(file_data, threshold=0.88):
    """近似重複検出（正規化テキストベース）"""
    duplicates = []
    group_id = 1
    
    # ファイル内容を読み込んで正規化
    normalized_content = {}
    for item in file_data:
        try:
            with open(item['path'], 'r', encoding='utf-8') as f:
                content = f.read()
                normalized_content[item['path']] = normalize_text(content)
        except:
            normalized_content[item['path']] = ""
    
    paths = list(file_data)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_a, path_b = paths[i]['path'], paths[j]['path']
            content_a = normalized_content.get(path_a, "")
            content_b = normalized_content.get(path_b, "")
            
            if content_a and content_b and len(content_a) > 100 and len(content_b) > 100:
                similarity = SequenceMatcher(None, content_a, content_b).ratio()
                
                if similarity >= threshold:
                    # サイズで優先度決定
                    preferred = path_a if paths[i]['size'] >= paths[j]['size'] else path_b
                    
                    duplicates.append({
                        'group_id': f'near_{group_id}',
                        'type': 'near-duplicate',
                        'path_a': path_a,
                        'path_b': path_b,
                        'similarity': round(similarity, 3),
                        'preferred': preferred
                    })
                    group_id += 1
    
    return duplicates


def detect_topic_duplicates(file_data, threshold=0.8):
    """トピック重複検出（キーワード・見出しベース）"""
    duplicates = []
    group_id = 1
    
    # キーワード抽出
    file_keywords = {}
    for item in file_data:
        try:
            with open(item['path'], 'r', encoding='utf-8') as f:
                content = f.read()
                keywords = extract_keywords(content, item['title'])
                file_keywords[item['path']] = keywords
        except:
            file_keywords[item['path']] = set()
    
    paths = list(file_data)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_a, path_b = paths[i]['path'], paths[j]['path']
            keywords_a = file_keywords.get(path_a, set())
            keywords_b = file_keywords.get(path_b, set())
            
            if keywords_a and keywords_b:
                intersection = keywords_a & keywords_b
                union = keywords_a | keywords_b
                
                if union:
                    similarity = len(intersection) / len(union)
                    
                    if similarity >= threshold:
                        # 最近更新されたファイルを優先
                        preferred = path_a if paths[i]['last_commit_date'] >= paths[j]['last_commit_date'] else path_b
                        
                        duplicates.append({
                            'group_id': f'topic_{group_id}',
                            'type': 'topic-duplicate',
                            'path_a': path_a,
                            'path_b': path_b,
                            'similarity': round(similarity, 3),
                            'preferred': preferred
                        })
                        group_id += 1
    
    return duplicates


def detect_multilingual_pairs(file_data):
    """多言語ペア検出（ja/enペア）"""
    pairs = []
    group_id = 1
    
    # 言語別グループ化
    ja_files = [item for item in file_data if item['lang'] == 'ja']
    en_files = [item for item in file_data if item['lang'] == 'en']
    
    for ja_file in ja_files:
        for en_file in en_files:
            # パス類似度チェック
            ja_path = Path(ja_file['path'])
            en_path = Path(en_file['path'])
            
            # ファイル名類似度
            ja_stem = ja_path.stem.lower()
            en_stem = en_path.stem.lower()
            
            # タイトル類似度（英数字のみで比較）
            ja_title_norm = re.sub(r'[^\w\s]', '', ja_file['title']).lower()
            en_title_norm = re.sub(r'[^\w\s]', '', en_file['title']).lower()
            
            # 類似度計算
            name_sim = SequenceMatcher(None, ja_stem, en_stem).ratio()
            title_sim = SequenceMatcher(None, ja_title_norm, en_title_norm).ratio()
            
            if name_sim > 0.7 or title_sim > 0.6:
                pairs.append({
                    'group_id': f'multilang_{group_id}',
                    'type': 'multilang-pair',
                    'path_a': ja_file['path'],
                    'path_b': en_file['path'],
                    'similarity': round(max(name_sim, title_sim), 3),
                    'preferred': 'both'  # 多言語ペアは両方保持
                })
                group_id += 1
    
    return pairs


def load_inventory(csv_path):
    """棚卸しCSVを読み込み"""
    file_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['size'] = int(row['size'])
            file_data.append(row)
    return file_data


def save_duplicates_csv(duplicates, output_path):
    """重複検出結果をCSVに保存"""
    fieldnames = ['group_id', 'type', 'path_a', 'path_b', 'similarity', 'preferred']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicates)
    
    print(f"📋 重複検出結果を {output_path} に保存")


def main():
    parser = argparse.ArgumentParser(description='Markdown重複検出ツール')
    parser.add_argument('--inventory', default='docs_inventory.csv', help='棚卸しCSVファイル')
    parser.add_argument('--out', default='md_duplicates.csv', help='重複検出結果CSV')
    
    args = parser.parse_args()
    
    print(f"📂 棚卸し結果 {args.inventory} を読み込み中...")
    file_data = load_inventory(args.inventory)
    
    print("🔍 重複検出を実行中...")
    all_duplicates = []
    
    # 完全重複検出
    exact_dups = detect_exact_duplicates(file_data)
    all_duplicates.extend(exact_dups)
    print(f"   完全重複: {len(exact_dups)} ペア")
    
    # 近似重複検出
    near_dups = detect_near_duplicates(file_data)
    all_duplicates.extend(near_dups)
    print(f"   近似重複: {len(near_dups)} ペア")
    
    # トピック重複検出
    topic_dups = detect_topic_duplicates(file_data)
    all_duplicates.extend(topic_dups)
    print(f"   トピック重複: {len(topic_dups)} ペア")
    
    # 多言語ペア検出
    multilang_pairs = detect_multilingual_pairs(file_data)
    all_duplicates.extend(multilang_pairs)
    print(f"   多言語ペア: {len(multilang_pairs)} ペア")
    
    save_duplicates_csv(all_duplicates, args.out)
    
    print(f"\n📊 重複検出サマリー:")
    print(f"   総検出ペア数: {len(all_duplicates)}")
    type_counts = defaultdict(int)
    for dup in all_duplicates:
        type_counts[dup['type']] += 1
    for dup_type, count in type_counts.items():
        print(f"   {dup_type}: {count}")


if __name__ == '__main__':
    main()