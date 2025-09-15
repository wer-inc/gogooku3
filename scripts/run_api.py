from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("gogooku3.api.server:app", host="0.0.0.0", port=8008, reload=False)


if __name__ == "__main__":
    main()

