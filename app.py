import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SERVER_FILE = PROJECT_DIR / "server.js"
PACKAGE_DIR = PROJECT_DIR / "node_modules"


def main() -> int:
    node_path = shutil.which("node")
    if not node_path:
        print("[ERROR] Node.js를 찾지 못했습니다. Node.js를 먼저 설치해 주세요.")
        return 1

    if not SERVER_FILE.exists():
        print(f"[ERROR] server.js 파일이 없습니다: {SERVER_FILE}")
        return 1

    if not PACKAGE_DIR.exists():
        print("[ERROR] node_modules가 없습니다. 먼저 아래 명령을 한 번 실행해 주세요.")
        print(f"        cd {PROJECT_DIR}")
        print("        npm.cmd install")
        return 1

    print("[SYSTEM] Union-web 서버를 시작합니다.")
    print("[SYSTEM] 브라우저에서 http://localhost:3000 으로 접속하세요.")
    print("[SYSTEM] 종료하려면 Ctrl+C 를 누르세요.")

    try:
        completed = subprocess.run(
            [node_path, str(SERVER_FILE)],
            cwd=PROJECT_DIR,
            check=False,
        )
        return completed.returncode
    except KeyboardInterrupt:
        print("\n[SYSTEM] 서버를 종료했습니다.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
