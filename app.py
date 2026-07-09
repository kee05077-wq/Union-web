from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_DIR / "requirements.txt"


def install_missing_packages(missing_package: str) -> None:
    print(f"[SYSTEM] Missing package detected: {missing_package}")
    print("[SYSTEM] Trying to install required Python packages automatically...")

    subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        cwd=PROJECT_DIR,
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
        cwd=PROJECT_DIR,
        check=True,
    )


try:
    from flask import Flask, jsonify, request, send_from_directory
    from openpyxl import Workbook, load_workbook
except ModuleNotFoundError as error:
    missing_package = getattr(error, "name", "required package")
    try:
        install_missing_packages(missing_package)
        print("[SYSTEM] Package installation completed. Restarting app...")
        subprocess.run([sys.executable, str(PROJECT_DIR / "app.py")], cwd=PROJECT_DIR, check=False)
        raise SystemExit(0)
    except Exception as install_error:
        print(f"[ERROR] Python package not found: {missing_package}")
        print(f"[ERROR] Automatic installation failed: {install_error}")
        print("[ERROR] Run the commands below in this folder:")
        print("        python -m ensurepip --upgrade")
        print("        python -m pip install -r requirements.txt")
        print("        python app.py")
        input("Press Enter to close...")
        raise SystemExit(1)
DATABASE_DIR = PROJECT_DIR / "database"
WORKBOOK_PATH = DATABASE_DIR / "users.xlsx"
SHEET_NAME = "Users"
HTML_FILES = {
    "index.html",
    "chatentry.html",
    "room.html",
    "story.html",
    "signup.html",
    "recover.html",
}

app = Flask(__name__, static_folder="static", static_url_path="/static")


def ensure_workbook() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    if WORKBOOK_PATH.exists():
        return

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = SHEET_NAME
    worksheet.append(["name", "birthDate", "username", "password", "createdAt"])
    workbook.save(WORKBOOK_PATH)


def read_users() -> list[dict[str, str]]:
    ensure_workbook()
    workbook = load_workbook(WORKBOOK_PATH)
    worksheet = workbook[SHEET_NAME]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()

    if not rows:
        return []

    headers = [str(value or "").strip() for value in rows[0]]
    users: list[dict[str, str]] = []

    for row in rows[1:]:
        if not any(row):
            continue
        users.append(
            {
                headers[index]: str(value or "").strip()
                for index, value in enumerate(row)
                if index < len(headers)
            }
        )

    return users


def write_users(users: list[dict[str, str]]) -> None:
    ensure_workbook()
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = SHEET_NAME
    headers = ["name", "birthDate", "username", "password", "createdAt"]
    worksheet.append(headers)

    for user in users:
        worksheet.append([user.get(header, "") for header in headers])

    workbook.save(WORKBOOK_PATH)
    workbook.close()


def same_text(left: str | None, right: str | None) -> bool:
    return str(left or "").strip() == str(right or "").strip()


@app.get("/")
def serve_index():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.get("/<path:filename>")
def serve_root_file(filename: str):
    if filename in HTML_FILES:
        return send_from_directory(PROJECT_DIR, filename)
    return send_from_directory(PROJECT_DIR, filename)


@app.post("/api/signup")
def signup():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    birth_date = payload.get("birthDate")
    username = payload.get("username")
    password = payload.get("password")

    if not all([name, birth_date, username, password]):
        return jsonify({"message": "모든 항목을 입력해 주세요."}), 400

    users = read_users()
    duplicate = any(same_text(user.get("username"), username) for user in users)

    if duplicate:
        return jsonify({"message": "이미 사용 중인 아이디입니다."}), 409

    users.append(
        {
            "name": str(name).strip(),
            "birthDate": str(birth_date).strip(),
            "username": str(username).strip(),
            "password": str(password).strip(),
            "createdAt": str(__import__("datetime").datetime.now().isoformat(timespec="seconds")),
        }
    )
    write_users(users)
    return jsonify({"message": "회원가입이 완료되었습니다."})


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username")
    password = payload.get("password")

    users = read_users()
    user = next(
        (
            item
            for item in users
            if same_text(item.get("username"), username) and same_text(item.get("password"), password)
        ),
        None,
    )

    if not user:
        return jsonify({"message": "아이디 또는 비밀번호가 일치하지 않습니다."}), 401

    return jsonify(
        {
            "message": f"{user['name']}님, 로그인되었습니다.",
            "profile": {
                "name": user["name"],
                "birthDate": user["birthDate"],
                "username": user["username"],
            },
        }
    )


@app.post("/api/find-id")
def find_id():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    birth_date = payload.get("birthDate")

    users = read_users()
    user = next(
        (
            item
            for item in users
            if same_text(item.get("name"), name) and same_text(item.get("birthDate"), birth_date)
        ),
        None,
    )

    if not user:
        return jsonify({"message": "일치하는 회원정보를 찾지 못했습니다."}), 404

    return jsonify({"username": user["username"]})


@app.post("/api/find-password")
def find_password():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username")
    name = payload.get("name")
    birth_date = payload.get("birthDate")

    users = read_users()
    user = next(
        (
            item
            for item in users
            if same_text(item.get("username"), username)
            and same_text(item.get("name"), name)
            and same_text(item.get("birthDate"), birth_date)
        ),
        None,
    )

    if not user:
        return jsonify({"message": "입력하신 정보와 일치하는 비밀번호를 찾지 못했습니다."}), 404

    return jsonify({"password": user["password"]})


if __name__ == "__main__":
    ensure_workbook()
    print("[SYSTEM] Union-web Flask 서버를 시작합니다.")
    print("[SYSTEM] 브라우저에서 http://localhost:3000 으로 접속하세요.")
    print("[SYSTEM] 종료하려면 Ctrl+C 를 누르세요.")
    app.run(host="0.0.0.0", port=5000, debug=False)
