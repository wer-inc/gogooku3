.PHONY: help health-check status

SCRIPT_DIR := $(CURDIR)

help:
	@echo "Tools targets"
	@echo "  make health-check   -> run shared diagnostics"
	@echo "  make status         -> summarize dataset/model state"

health-check:
	@"$(SCRIPT_DIR)/health-check.sh"

status:
	@"$(SCRIPT_DIR)/codex.sh" --status
