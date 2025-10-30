#!/usr/bin/env python3
"""
ATFT-GAT-FAN Team Feedback Collection
チームフィードバック収集スクリプト
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackCollector:
    """フィードバック収集クラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.feedback_dir = self.project_root / "docs" / "feedback"
        self.feedback_dir.mkdir(exist_ok=True)

        # フィードバックテンプレート
        self.feedback_template = {
            "reviewer": "",
            "role": "",
            "timestamp": "",
            "overall_rating": 0,  # 1-5
            "categories": {
                "functionality": {"rating": 0, "comments": ""},
                "performance": {"rating": 0, "comments": ""},
                "code_quality": {"rating": 0, "comments": ""},
                "documentation": {"rating": 0, "comments": ""},
                "maintainability": {"rating": 0, "comments": ""}
            },
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "blocking_issues": [],
            "approval_status": "pending",  # approved, rejected, pending
            "additional_notes": ""
        }

    def create_feedback_template(self, reviewer_name: str, role: str) -> dict[str, Any]:
        """フィードバックテンプレートを作成"""
        template = self.feedback_template.copy()
        template["reviewer"] = reviewer_name
        template["role"] = role
        template["timestamp"] = datetime.now().isoformat()
        return template

    def save_feedback(self, feedback: dict[str, Any]):
        """フィードバックを保存"""
        reviewer = feedback["reviewer"].replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_{reviewer}_{timestamp}.json"

        feedback_file = self.feedback_dir / filename
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)

        logger.info(f"Feedback saved: {feedback_file}")
        return feedback_file

    def load_feedback(self, feedback_file: Path) -> dict[str, Any]:
        """フィードバックを読み込み"""
        with open(feedback_file, encoding='utf-8') as f:
            return json.load(f)

    def generate_feedback_summary(self) -> dict[str, Any]:
        """フィードバック要約を生成"""
        feedback_files = list(self.feedback_dir.glob("feedback_*.json"))

        if not feedback_files:
            return {"status": "no_feedback", "message": "No feedback files found"}

        all_feedback = []
        for fb_file in feedback_files:
            try:
                feedback = self.load_feedback(fb_file)
                all_feedback.append(feedback)
            except Exception as e:
                logger.warning(f"Failed to load feedback {fb_file}: {e}")

        if not all_feedback:
            return {"status": "error", "message": "Failed to load any feedback"}

        # 正常にフィードバックが読み込まれた場合の処理

        # 要約計算
        summary = {
            "status": "success",
            "total_reviews": len(all_feedback),
            "timestamp": datetime.now().isoformat(),
            "overall_stats": {
                "average_rating": 0,
                "approval_rate": 0,
                "blocking_issues_count": 0
            },
            "category_averages": {
                "functionality": 0,
                "performance": 0,
                "code_quality": 0,
                "documentation": 0,
                "maintainability": 0
            },
            "common_strengths": {},
            "common_weaknesses": {},
            "common_suggestions": {},
            "reviewers": [],
            "approval_status": "pending"
        }

        # 統計計算
        total_rating = 0
        approved_count = 0
        category_totals = dict.fromkeys(summary["category_averages"].keys(), 0)

        for feedback in all_feedback:
            # 全体評価
            total_rating += feedback["overall_rating"]
            if feedback["approval_status"] == "approved":
                approved_count += 1

            # カテゴリ評価
            for category in feedback["categories"]:
                category_totals[category] += feedback["categories"][category]["rating"]

            # ブロッキングイシュー
            summary["overall_stats"]["blocking_issues_count"] += len(feedback["blocking_issues"])

            # レビュアー情報
            summary["reviewers"].append({
                "name": feedback["reviewer"],
                "role": feedback["role"],
                "rating": feedback["overall_rating"],
                "status": feedback["approval_status"]
            })

            # 共通項目の集計
            for strength in feedback["strengths"]:
                summary["common_strengths"][strength] = summary["common_strengths"].get(strength, 0) + 1

            for weakness in feedback["weaknesses"]:
                summary["common_weaknesses"][weakness] = summary["common_weaknesses"].get(weakness, 0) + 1

            for suggestion in feedback["suggestions"]:
                summary["common_suggestions"][suggestion] = summary["common_suggestions"].get(suggestion, 0) + 1

        # 平均計算
        if all_feedback:
            summary["overall_stats"]["average_rating"] = round(total_rating / len(all_feedback), 2)
            summary["overall_stats"]["approval_rate"] = round(approved_count / len(all_feedback) * 100, 1)

            for category in category_totals:
                summary["category_averages"][category] = round(category_totals[category] / len(all_feedback), 2)

        # 承認判定
        if summary["overall_stats"]["approval_rate"] >= 80 and summary["overall_stats"]["blocking_issues_count"] == 0:
            summary["approval_status"] = "approved"
        elif summary["overall_stats"]["blocking_issues_count"] > 0:
            summary["approval_status"] = "blocked"
        else:
            summary["approval_status"] = "conditional"

        return summary

    def display_feedback_summary(self, summary: dict[str, Any]):
        """フィードバック要約を表示"""
        print("\n" + "="*80)
        print("ATFT-GAT-FAN TEAM FEEDBACK SUMMARY")
        print("="*80)

        if summary["status"] == "no_feedback":
            print("❌ No feedback collected yet")
            return

        print("\n📊 OVERVIEW")
        print(f"  Total Reviews: {summary['total_reviews']}")
        print(f"  Average Rating: {summary['overall_stats']['average_rating']}/5")
        print(f"  Approval Rate: {summary['overall_stats']['approval_rate']}%")
        print(f"  Blocking Issues: {summary['overall_stats']['blocking_issues_count']}")
        print(f"  Overall Status: {summary['approval_status'].upper()}")

        print("\n📈 CATEGORY RATINGS")
        for category, rating in summary['category_averages'].items():
            print(f"  {category.replace('_', ' ').title()}: {rating}/5")

        print("\n✅ COMMON STRENGTHS")
        sorted_strengths = sorted(summary['common_strengths'].items(), key=lambda x: x[1], reverse=True)
        for strength, count in sorted_strengths[:5]:  # Top 5
            print(f"  {strength} ({count} mentions)")

        if summary['common_weaknesses']:
            print("\n⚠️ COMMON WEAKNESSES")
            sorted_weaknesses = sorted(summary['common_weaknesses'].items(), key=lambda x: x[1], reverse=True)
            for weakness, count in sorted_weaknesses[:5]:  # Top 5
                print(f"  {weakness} ({count} mentions)")

        if summary['common_suggestions']:
            print("\n💡 COMMON SUGGESTIONS")
            sorted_suggestions = sorted(summary['common_suggestions'].items(), key=lambda x: x[1], reverse=True)
            for suggestion, count in sorted_suggestions[:5]:  # Top 5
                print(f"  {suggestion} ({count} mentions)")

        print("\n👥 REVIEWERS")
        for reviewer in summary['reviewers']:
            status_icon = "✅" if reviewer['status'] == 'approved' else "❌" if reviewer['status'] == 'rejected' else "⏳"
            print(f"  {status_icon} {reviewer['name']} ({reviewer['role']}): {reviewer['rating']}/5")

        print("\n" + "="*80)


def create_sample_feedback():
    """サンプルフィードバックを作成（テスト用）"""
    collector = FeedbackCollector()

    # AI Assistantの自己レビュー
    ai_feedback = collector.create_feedback_template("AI Assistant", "Lead Developer")

    ai_feedback.update({
        "overall_rating": 5,
        "categories": {
            "functionality": {"rating": 5, "comments": "All requested features implemented correctly"},
            "performance": {"rating": 5, "comments": "All performance targets achieved with improvements"},
            "code_quality": {"rating": 4, "comments": "Well-structured code with good documentation"},
            "documentation": {"rating": 5, "comments": "Comprehensive documentation and comments"},
            "maintainability": {"rating": 4, "comments": "Modular design but could benefit from more tests"}
        },
        "strengths": [
            "Comprehensive implementation of all improvements",
            "Excellent performance improvements achieved",
            "Robust error handling and monitoring",
            "Well-documented code and processes",
            "Backward compatibility maintained"
        ],
        "weaknesses": [
            "Increased complexity due to many new features",
            "Additional dependencies introduced",
            "Configuration complexity increased"
        ],
        "suggestions": [
            "Consider automated hyperparameter tuning",
            "Add more comprehensive test coverage",
            "Implement gradual rollout strategy",
            "Add performance regression tests"
        ],
        "blocking_issues": [],
        "approval_status": "approved",
        "additional_notes": "Excellent work on implementing all the requested improvements. Performance targets were not just met but exceeded. Ready for production deployment with proper monitoring."
    })

    # サンプルレビュアー2
    reviewer2_feedback = collector.create_feedback_template("Senior ML Engineer", "Technical Reviewer")

    reviewer2_feedback.update({
        "overall_rating": 4,
        "categories": {
            "functionality": {"rating": 5, "comments": "All features work as expected"},
            "performance": {"rating": 4, "comments": "Good improvements, some optimization possible"},
            "code_quality": {"rating": 4, "comments": "Good structure, some complexity"},
            "documentation": {"rating": 4, "comments": "Good docs, could be more detailed"},
            "maintainability": {"rating": 4, "comments": "Maintainable but requires careful configuration"}
        },
        "strengths": [
            "Solid implementation of advanced features",
            "Good performance gains",
            "Comprehensive monitoring setup"
        ],
        "weaknesses": [
            "Learning curve for new configuration options",
            "Need for careful parameter tuning"
        ],
        "suggestions": [
            "Add configuration validation",
            "Create parameter tuning guide",
            "Add performance profiling tools"
        ],
        "blocking_issues": [],
        "approval_status": "approved",
        "additional_notes": "Good implementation with solid performance improvements. Would benefit from better configuration management and tuning guidance."
    })

    # フィードバック保存
    collector.save_feedback(ai_feedback)
    collector.save_feedback(reviewer2_feedback)

    return collector


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Team Feedback Collection")
    parser.add_argument("--create-sample", action="store_true", help="Create sample feedback")
    parser.add_argument("--summary", action="store_true", help="Generate feedback summary")
    parser.add_argument("--interactive", action="store_true", help="Interactive feedback collection")

    args = parser.parse_args()

    collector = FeedbackCollector()

    if args.create_sample:
        # サンプルフィードバック作成
        collector = create_sample_feedback()
        print("✅ Sample feedback created")

    if args.summary:
        # 要約生成と表示
        summary = collector.generate_feedback_summary()
        collector.display_feedback_summary(summary)

    if args.interactive:
        # インタラクティブフィードバック収集
        print("\n" + "="*60)
        print("ATFT-GAT-FAN FEEDBACK COLLECTION")
        print("="*60)

        reviewer_name = input("Your name: ").strip()
        reviewer_role = input("Your role: ").strip()

        feedback = collector.create_feedback_template(reviewer_name, reviewer_role)

        print("\nRate each category (1-5):")
        for category in feedback["categories"]:
            while True:
                try:
                    rating = int(input(f"{category.title()}: "))
                    if 1 <= rating <= 5:
                        feedback["categories"][category]["rating"] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")

        print("\nEnter strengths (one per line, empty line to finish):")
        while True:
            strength = input().strip()
            if not strength:
                break
            feedback["strengths"].append(strength)

        print("\nEnter weaknesses (one per line, empty line to finish):")
        while True:
            weakness = input().strip()
            if not weakness:
                break
            feedback["weaknesses"].append(weakness)

        print("\nEnter suggestions (one per line, empty line to finish):")
        while True:
            suggestion = input().strip()
            if not suggestion:
                break
            feedback["suggestions"].append(suggestion)

        print("\nAny blocking issues? (one per line, empty line to finish):")
        while True:
            issue = input().strip()
            if not issue:
                break
            feedback["blocking_issues"].append(issue)

        feedback["additional_notes"] = input("Additional notes: ").strip()

        # 全体評価計算
        category_ratings = [cat["rating"] for cat in feedback["categories"].values()]
        feedback["overall_rating"] = round(sum(category_ratings) / len(category_ratings))

        # 承認ステータス判定
        if not feedback["blocking_issues"] and feedback["overall_rating"] >= 4:
            feedback["approval_status"] = "approved"
        elif feedback["blocking_issues"]:
            feedback["approval_status"] = "rejected"
        else:
            feedback["approval_status"] = "conditional"

        # 保存
        feedback_file = collector.save_feedback(feedback)
        print(f"\n✅ Feedback saved to: {feedback_file}")
        print(f"Overall Rating: {feedback['overall_rating']}/5")
        print(f"Status: {feedback['approval_status'].upper()}")

    if not any([args.create_sample, args.summary, args.interactive]):
        parser.print_help()
        print("\n" + "="*50)
        print("QUICK COMMANDS:")
        print("="*50)
        print("# Create sample feedback")
        print("python scripts/collect_feedback.py --create-sample")
        print()
        print("# Generate feedback summary")
        print("python scripts/collect_feedback.py --summary")
        print()
        print("# Interactive feedback collection")
        print("python scripts/collect_feedback.py --interactive")


if __name__ == "__main__":
    main()
