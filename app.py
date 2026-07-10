from __future__ import annotations

import subprocess
import sys
import time
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_DIR / "requirements.txt"

def install_missing_packages(missing_package: str) -> None:
    print(f"[SYSTEM] Missing package detected: {missing_package}")
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=PROJECT_DIR, check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)], cwd=PROJECT_DIR, check=True)

try:
    from flask import Flask, jsonify, request, send_from_directory, render_template
    from flask_socketio import SocketIO, emit, join_room, leave_room
    from openpyxl import Workbook, load_workbook
    import base64
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from unicode import join_jamos
except ModuleNotFoundError as error:
    missing_package = getattr(error, "name", "required package")
    try:
        install_missing_packages(missing_package)
        subprocess.run([sys.executable, str(PROJECT_DIR / "app.py")], cwd=PROJECT_DIR, check=False)
        raise SystemExit(0)
    except Exception as install_error:
        print(f"[ERROR] Python package not found: {missing_package}")
        input("Press Enter to close...")
        raise SystemExit(1)

# ==========================================
# 기본 설정
# ==========================================
DATABASE_DIR = PROJECT_DIR / "database"
WORKBOOK_PATH = DATABASE_DIR / "users.xlsx"
SHEET_NAME = "Users"
HTML_FILES = {"index.html", "chatentry.html", "room.html", "story.html", "signup.html", "recover.html", "translate.html", "education.html"}

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['SECRET_KEY'] = 'union-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=1e7, async_mode='eventlet')

# 수어 큐 및 타임아웃 관리
user_finger_queues = {}
user_last_active = {}

# 정규식 필터 (단일 자모 제거)
def remove_single_letters(word: str) -> str:
    return re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', '', word)

# ==========================================
# YOLO 모델 로드
# ==========================================
try:
    print("[SYSTEM] 커스텀 수어 YOLO 모델(best.pt)을 로드합니다...")
    model = YOLO('best.pt') 
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")

CLASS_MAP = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 
    8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 
    16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 
    24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅢ', 29: 'ㅚ', 30: 'ㅟ'
}

# ==========================================
# 엑셀 DB 함수
# ==========================================
def ensure_workbook() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    if WORKBOOK_PATH.exists(): return
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
    if not rows: return []
    headers = [str(value or "").strip() for value in rows[0]]
    users: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(row): continue
        users.append({headers[index]: str(value or "").strip() for index, value in enumerate(row) if index < len(headers)})
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

# ==========================================
# 라우터 및 API
# ==========================================
@app.get("/")
def serve_index(): return render_template("index.html")

@app.get("/<path:filename>")
def serve_root_file(filename: str):
    if filename in HTML_FILES: return render_template(filename)
    return send_from_directory(PROJECT_DIR, filename)

@app.post("/api/signup")
def signup():
    payload = request.get_json(silent=True) or {}
    name, birth_date, username, password = payload.get("name"), payload.get("birthDate"), payload.get("username"), payload.get("password")
    if not all([name, birth_date, username, password]): return jsonify({"message": "모든 항목을 입력해 주세요."}), 400
    users = read_users()
    if any(same_text(user.get("username"), username) for user in users): return jsonify({"message": "이미 사용 중인 아이디입니다."}), 409
    users.append({"name": str(name).strip(), "birthDate": str(birth_date).strip(), "username": str(username).strip(), "password": str(password).strip(), "createdAt": str(__import__("datetime").datetime.now().isoformat(timespec="seconds"))})
    write_users(users)
    return jsonify({"message": "회원가입이 완료되었습니다."})

@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username, password = payload.get("username"), payload.get("password")
    users = read_users()
    user = next((item for item in users if same_text(item.get("username"), username) and same_text(item.get("password"), password)), None)
    if not user: return jsonify({"message": "아이디 또는 비밀번호가 일치하지 않습니다."}), 401
    return jsonify({"message": f"{user['name']}님, 로그인되었습니다.", "profile": {"name": user["name"], "birthDate": user["birthDate"], "username": user["username"]}})

@app.post("/api/find-id")
def find_id():
    payload = request.get_json(silent=True) or {}
    name, birth_date = payload.get("name"), payload.get("birthDate")
    users = read_users()
    user = next((item for item in users if same_text(item.get("name"), name) and same_text(item.get("birthDate"), birth_date)), None)
    if not user: return jsonify({"message": "일치하는 회원정보를 찾지 못했습니다."}), 404
    return jsonify({"username": user["username"]})

@app.post("/api/find-password")
def find_password():
    payload = request.get_json(silent=True) or {}
    username, name, birth_date = payload.get("username"), payload.get("name"), payload.get("birthDate")
    users = read_users()
    user = next((item for item in users if same_text(item.get("username"), username) and same_text(item.get("name"), name) and same_text(item.get("birthDate"), birth_date)), None)
    if not user: return jsonify({"message": "입력하신 정보와 일치하는 비밀번호를 찾지 못했습니다."}), 404
    return jsonify({"password": user["password"]})

# ==========================================
# 웹소켓 (채팅, 수어, WebRTC)
# ==========================================
@socketio.on('join_room')
def on_join(data):
    room = data['room']
    name = data['name']
    join_room(room)
    emit('user_joined', {'name': name, 'sid': request.sid}, to=room, include_self=False)
    emit('receive_message', {'name': 'SYSTEM', 'msg': f'{name}님이 회의실에 입장하셨습니다.'}, to=room, include_self=True)

@socketio.on('send_message')
def on_chat_message(data): 
    emit('receive_message', data, to=data['room'], include_self=True)

@socketio.on('send_file')
def on_file_upload(data): 
    emit('receive_file', data, to=data['room'], include_self=True)

@socketio.on('webrtc_offer')
def on_webrtc_offer(data): emit('webrtc_offer', data, to=data['room'], include_self=False)

@socketio.on('webrtc_answer')
def on_webrtc_answer(data): emit('webrtc_answer', data, to=data['room'], include_self=False)

@socketio.on('webrtc_ice')
def on_webrtc_ice(data): emit('webrtc_ice', data, to=data['room'], include_self=False)

@socketio.on('sign_frame')
def on_sign_frame(data):
    room = data['room']
    name = data['name']
    image_b64 = data['image'].split(',')[1] 
    
    img_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img, verbose=False)
    predicted_jamo = None
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        class_id = int(best_box.cls[0].item())
        confidence = best_box.conf[0].item()
        
        # 신뢰도 0.7 이상만 통과 (노이즈 방어)
        if confidence > 0.7:  
            predicted_jamo = CLASS_MAP.get(class_id)

    queue_key = f"{room}_{name}"
    current_time = time.time()
    
    if queue_key not in user_finger_queues: 
        user_finger_queues[queue_key] = []
        user_last_active[queue_key] = current_time
        
    queue = user_finger_queues[queue_key]

    # 1. 입력 중 (1.5초 대기 갱신)
    if predicted_jamo is not None:
        user_last_active[queue_key] = current_time
        
        if len(queue) == 0 or queue[-1] != predicted_jamo:
            queue.append(predicted_jamo)
            
            raw_assembled = join_jamos("".join(queue), ignore_err=True)
            filtered_preview = remove_single_letters(raw_assembled)
            
            display_text = filtered_preview if filtered_preview else raw_assembled
            
            emit('sign_progress', {
                'current_jamo': predicted_jamo, 
                'text': display_text
            }, to=request.sid)
                
    # 2. 입력 중단 시 (1.5초 타임아웃 판정)
    else:
        if len(queue) > 0:
            time_passed = current_time - user_last_active.get(queue_key, current_time)
            
            if time_passed >= 1.5:
                raw_assembled = join_jamos("".join(queue), ignore_err=True)
                filtered_result = remove_single_letters(raw_assembled)
                
                # 완성된 문장만 송출
                if filtered_result.strip():
                    emit('sign_result', {'name': name, 'text': filtered_result}, to=room, include_self=True)
                    
                user_finger_queues[queue_key] = []
                emit('sign_progress', {'current_jamo': '', 'text': ''}, to=request.sid)

if __name__ == "__main__":
    ensure_workbook()
    print("[SYSTEM] Union-web 실시간 소켓 서버를 시작합니다.")
    socketio.run(app, host="0.0.0.0", port=80, debug=False)
