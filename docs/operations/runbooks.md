# 🚧 運用手順書 (作成中)

このドキュメントは現在作成中です。

## 📋 予定内容

- **標準運用手順**: システム起動・停止・定期メンテナンス
- **チェックリスト**: 運用作業の確認項目
- **監視項目**: 確認すべきメトリクス・アラート
- **障害対応**: エスカレーション手順・連絡先

## 🔄 現在の運用方法

暫定的な運用手順は以下をご参照ください：

### 基本操作
```bash
# システム起動
systemctl start gogooku3.service

# 学習パイプライン実行
gogooku3 train --config configs/atft/train/production.yaml

# システム停止
systemctl stop gogooku3.service
```

### 監視・確認
```bash
# システム状態確認
systemctl status gogooku3.service
journalctl -u gogooku3.service --since "10 minutes ago"

# メモリ・CPU使用量確認
nvidia-smi
htop
```

詳細は [はじめに](../getting-started.md) をご参照ください。

## 📞 お問い合わせ

運用に関するご質問は以下まで：
- [GitHub Issues](https://github.com/your-org/gogooku3/issues)
- [FAQ](../faq.md) - よくある質問

---

**🚧 作成予定日**: 2025年9月  
**👥 担当**: DevOps チーム
## Regenerate ml_dataset_20250827_090410.csv
Original path: `output/ml_dataset_20250827_090410.csv`
To regenerate, run the appropriate script or pipeline.


## Regenerate ml_dataset_latest.parquet
Original path: `output/ml_dataset_latest.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate best_practices_report_20250827_142751.json
Original path: `output/best_practices_report_20250827_142751.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate ml_dataset_latest_metadata.json
Original path: `output/ml_dataset_latest_metadata.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate ml_dataset_20250827_085629_metadata.json
Original path: `output/ml_dataset_20250827_085629_metadata.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate ml_dataset_20250827_175915_metadata.json
Original path: `output/ml_dataset_20250827_175915_metadata.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate ml_dataset_20250827_174908_metadata.json
Original path: `output/ml_dataset_20250827_174908_metadata.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate deduplication_report_20250828_145722.json
Original path: `output/deduplication_report_20250828_145722.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate atft_training_data.parquet
Original path: `output/atft_training_data.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate .gitkeep
Original path: `output/.gitkeep`
To regenerate, run the appropriate script or pipeline.


## Regenerate complete_training_report_20250827_135507.md
Original path: `output/results/complete_training_report_20250827_135507.md`
To regenerate, run the appropriate script or pipeline.


## Regenerate complete_training_result_20250827_135329.json
Original path: `output/results/complete_training_result_20250827_135329.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate test_1395.parquet
Original path: `output/atft_data/test_1395.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate val_1356.parquet
Original path: `output/atft_data/val_1356.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate train_1347.parquet
Original path: `output/atft_data/train_1347.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate metadata.json
Original path: `output/atft_data/metadata.json`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1376.parquet
Original path: `output/atft_data/val/1376.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1343.parquet
Original path: `output/atft_data/val/1343.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1350.parquet
Original path: `output/atft_data/val/1350.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1326.parquet
Original path: `output/atft_data/val/1326.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1317.parquet
Original path: `output/atft_data/val/1317.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1387.parquet
Original path: `output/atft_data/val/1387.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1385.parquet
Original path: `output/atft_data/val/1385.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1319.parquet
Original path: `output/atft_data/val/1319.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1303.parquet
Original path: `output/atft_data/val/1303.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1322.parquet
Original path: `output/atft_data/val/1322.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1341.parquet
Original path: `output/atft_data/val/1341.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1312.parquet
Original path: `output/atft_data/val/1312.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1361.parquet
Original path: `output/atft_data/val/1361.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1378.parquet
Original path: `output/atft_data/val/1378.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1395.parquet
Original path: `output/atft_data/val/1395.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1347.parquet
Original path: `output/atft_data/val/1347.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1336.parquet
Original path: `output/atft_data/val/1336.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1310.parquet
Original path: `output/atft_data/val/1310.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1315.parquet
Original path: `output/atft_data/val/1315.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1314.parquet
Original path: `output/atft_data/val/1314.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1374.parquet
Original path: `output/atft_data/val/1374.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1334.parquet
Original path: `output/atft_data/val/1334.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1367.parquet
Original path: `output/atft_data/val/1367.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1399.parquet
Original path: `output/atft_data/val/1399.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1307.parquet
Original path: `output/atft_data/val/1307.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1398.parquet
Original path: `output/atft_data/val/1398.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1338.parquet
Original path: `output/atft_data/val/1338.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1333.parquet
Original path: `output/atft_data/val/1333.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1349.parquet
Original path: `output/atft_data/val/1349.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1327.parquet
Original path: `output/atft_data/val/1327.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1388.parquet
Original path: `output/atft_data/val/1388.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1330.parquet
Original path: `output/atft_data/val/1330.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1309.parquet
Original path: `output/atft_data/val/1309.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1313.parquet
Original path: `output/atft_data/val/1313.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1364.parquet
Original path: `output/atft_data/val/1364.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1362.parquet
Original path: `output/atft_data/val/1362.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1356.parquet
Original path: `output/atft_data/val/1356.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1354.parquet
Original path: `output/atft_data/val/1354.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1311.parquet
Original path: `output/atft_data/val/1311.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1370.parquet
Original path: `output/atft_data/val/1370.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1342.parquet
Original path: `output/atft_data/val/1342.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1357.parquet
Original path: `output/atft_data/val/1357.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1320.parquet
Original path: `output/atft_data/val/1320.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1345.parquet
Original path: `output/atft_data/val/1345.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1391.parquet
Original path: `output/atft_data/val/1391.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1325.parquet
Original path: `output/atft_data/val/1325.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1323.parquet
Original path: `output/atft_data/val/1323.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1329.parquet
Original path: `output/atft_data/val/1329.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1392.parquet
Original path: `output/atft_data/val/1392.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1302.parquet
Original path: `output/atft_data/val/1302.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1337.parquet
Original path: `output/atft_data/val/1337.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1386.parquet
Original path: `output/atft_data/val/1386.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1339.parquet
Original path: `output/atft_data/val/1339.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1335.parquet
Original path: `output/atft_data/val/1335.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1332.parquet
Original path: `output/atft_data/val/1332.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1393.parquet
Original path: `output/atft_data/val/1393.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1301.parquet
Original path: `output/atft_data/val/1301.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1348.parquet
Original path: `output/atft_data/val/1348.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1372.parquet
Original path: `output/atft_data/val/1372.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1380.parquet
Original path: `output/atft_data/val/1380.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1397.parquet
Original path: `output/atft_data/val/1397.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1371.parquet
Original path: `output/atft_data/val/1371.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1366.parquet
Original path: `output/atft_data/val/1366.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1384.parquet
Original path: `output/atft_data/val/1384.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1390.parquet
Original path: `output/atft_data/val/1390.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1400.parquet
Original path: `output/atft_data/val/1400.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1321.parquet
Original path: `output/atft_data/val/1321.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1352.parquet
Original path: `output/atft_data/val/1352.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1306.parquet
Original path: `output/atft_data/val/1306.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1383.parquet
Original path: `output/atft_data/val/1383.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1331.parquet
Original path: `output/atft_data/val/1331.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1328.parquet
Original path: `output/atft_data/val/1328.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1360.parquet
Original path: `output/atft_data/val/1360.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1358.parquet
Original path: `output/atft_data/val/1358.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1382.parquet
Original path: `output/atft_data/val/1382.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1304.parquet
Original path: `output/atft_data/val/1304.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1324.parquet
Original path: `output/atft_data/val/1324.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1308.parquet
Original path: `output/atft_data/val/1308.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1344.parquet
Original path: `output/atft_data/val/1344.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1368.parquet
Original path: `output/atft_data/val/1368.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1379.parquet
Original path: `output/atft_data/val/1379.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1340.parquet
Original path: `output/atft_data/val/1340.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1351.parquet
Original path: `output/atft_data/val/1351.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1346.parquet
Original path: `output/atft_data/val/1346.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1316.parquet
Original path: `output/atft_data/val/1316.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1355.parquet
Original path: `output/atft_data/val/1355.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1377.parquet
Original path: `output/atft_data/val/1377.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1365.parquet
Original path: `output/atft_data/val/1365.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1373.parquet
Original path: `output/atft_data/val/1373.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1375.parquet
Original path: `output/atft_data/val/1375.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1381.parquet
Original path: `output/atft_data/val/1381.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1353.parquet
Original path: `output/atft_data/val/1353.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1396.parquet
Original path: `output/atft_data/val/1396.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1394.parquet
Original path: `output/atft_data/val/1394.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1369.parquet
Original path: `output/atft_data/val/1369.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1359.parquet
Original path: `output/atft_data/val/1359.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1318.parquet
Original path: `output/atft_data/val/1318.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1389.parquet
Original path: `output/atft_data/val/1389.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1305.parquet
Original path: `output/atft_data/val/1305.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate 1363.parquet
Original path: `output/atft_data/val/1363.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate STOCK_0001.parquet
Original path: `output/atft_data/train/STOCK_0001.parquet`
To regenerate, run the appropriate script or pipeline.


## Regenerate data_metadata_20250827_145519.json
Original path: `output/metadata/data_metadata_20250827_145519.json`
To regenerate, run the appropriate script or pipeline.
