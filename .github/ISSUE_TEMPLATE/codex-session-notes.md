name: Codex Session Notes
description: Track open tasks and context between Codex sessions.
title: "[Codex] YYYY-MM-DD session notes"
labels: ["codex", "todo"]
body:
  - type: textarea
    id: pending
    attributes:
      label: Pending tasks
      description: 次回に持ち越す作業を箇条書きで記入してください。
      placeholder: "- [ ] Run safe training on latest dataset\n- [ ] Update docs with new features"
    validations:
      required: true
  - type: textarea
    id: context
    attributes:
      label: Context / Links
      description: 関連する PR やレポート、ログなどがあれば記載してください。
      placeholder: "- PR: #1234\n- Dataset report: docs/reports/safe_training_20250918.md"
  - type: textarea
    id: notes
    attributes:
      label: Free-form notes
      description: その他メモ（調査結果、注意点など）
