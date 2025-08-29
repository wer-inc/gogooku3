# 🔧 トラブルシューティング (作成中)

このドキュメントは現在作成中です。

## 📋 予定内容

- **典型的な障害と対処法**: システム障害・データ取得エラー・学習失敗等
- **復旧手順**: 段階的な復旧プロセス
- **ログ解析**: エラーログの読み方・分析方法
- **緊急時対応**: エスカレーション・連絡体制

## 🚨 現在の緊急時対応

暫定的なトラブルシューティング情報：

### よくある問題と対処

**1. メモリ不足エラー**
```bash
# メモリ制限での実行
gogooku3 train --memory-limit 4

# または詳細情報
python scripts/run_safe_training.py --memory-limit 4 --verbose
```

**2. GPU学習エラー**
```bash
# CPU学習への切り替え
gogooku3 train --accelerator cpu

# GPU状態確認
nvidia-smi
```

**3. データ取得失敗**
```bash
# JQuants接続確認
python -c "from gogooku3 import settings; print(settings.jquants_email)"

# .env設定確認
cat .env | grep JQUANTS
```

### ログ確認方法
```bash
# システムログ
make logs

# アプリケーションログ
tail -f logs/gogooku3.log

# エラー詳細
python scripts/run_safe_training.py --verbose --n-splits 1
```

## 📚 暫定リソース

詳細なトラブルシューティング情報は以下をご参照：

- [FAQ](../faq.md) - よくある質問と解決方法
- [Getting Started](../getting-started.md) - 基本的なセットアップ問題
- [GitHub Issues](https://github.com/your-org/gogooku3/issues) - 既知の問題・バグ報告

## 📞 緊急時連絡先

- **システム障害**: admin@example.com
- **技術的質問**: GitHub Issues
- **設定・運用**: [FAQ](../faq.md) を先にご確認ください

---

**🚧 作成予定日**: 2025年9月  
**👥 担当**: DevOps・開発チーム