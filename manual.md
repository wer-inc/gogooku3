# Quick Start

```bash
# 1) Build dataset in background (recommended; logs + PID/PGID saved)
make dataset-bg
# Monitor: tail -f _logs/dataset/*.log
# Stop:    kill <PID>  or  kill -TERM -<PGID>

# 2) Train a stable model
make train
make train-optimized
```
