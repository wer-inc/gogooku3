.PHONY: help dataset train-atft train-apex health-check status

help:
	@echo "gogooku5 migration helper targets"
	@echo "  make dataset        -> delegate to data package"
	@echo "  make train-atft     -> delegate to ATFT-GAT-FAN package"
	@echo "  make train-apex     -> delegate to APEX-Ranker package"
	@echo "  make health-check   -> run shared health diagnostics"
	@echo "  make status         -> summarize dataset + model states"

_dataset_make := $(CURDIR)/data/Makefile
_models_atft := $(CURDIR)/models/atft_gat_fan/Makefile.train
_models_apex := $(CURDIR)/models/apex_ranker/Makefile.train
_tools_health := $(CURDIR)/tools/health-check.sh

ifeq (,$(wildcard $(_dataset_make)))
$(warning data package Makefile not found yet; add gogooku5/data/Makefile)
endif
ifeq (,$(wildcard $(_models_atft)))
$(warning ATFT-GAT-FAN Makefile.train not found yet; add gogooku5/models/atft_gat_fan/Makefile.train)
endif
ifeq (,$(wildcard $(_models_apex)))
$(warning APEX-Ranker Makefile.train not found yet; add gogooku5/models/apex_ranker/Makefile.train)
endif
ifeq (,$(wildcard $(_tools_health)))
$(warning health-check script not found yet; add gogooku5/tools/health-check.sh)
endif

# Delegated targets -------------------------------------------------------

dataset:
	@$(MAKE) -C data build

train-atft:
	@$(MAKE) -C models/atft_gat_fan train

train-apex:
	@$(MAKE) -C models/apex_ranker train

health-check:
	@$(MAKE) -C tools health-check

status:
	@$(MAKE) -C tools status
