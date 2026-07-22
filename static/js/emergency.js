// 긴급신고 페이지 로직 (초안)
// 카메라 수어 인식은 room_ui.js와 동일한 소켓 파이프라인(sign_frame -> sign_progress/sign_result)을
// 그대로 재사용합니다. 나 혼자만 있는 임시 방(room)을 만들어 인식 결과만 받습니다.
//
// 112/119로 텍스트를 실제로 "자동 전송"하는 공개 API는 없습니다(브라우저가 사용자 동의 없이
// 문자를 보내는 것 자체가 보안상 불가능/금지된 동작). 대신 sms: 링크로 112/119 앞으로
// 신고 문구 + 위치가 미리 채워진 문자 작성창을 열어주고, 마지막 "전송"만 사용자가 직접 누릅니다.

const socket = io();
const ROOM_NAME = `emergency-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const USER_NAME = '긴급신고자';

let myStream = null;
let captureInterval = null;
let signOn = false;
let joined = false;
let locationWatchId = null;

const cameraStage = document.getElementById('cameraStage');
const myFace = document.getElementById('myFace');
const camOffHint = document.getElementById('camOffHint');
const livePreview = document.getElementById('livePreview');
const statusBadge = document.getElementById('statusBadge');
const toggleBtn = document.getElementById('toggleBtn');
const reportText = document.getElementById('reportText');
const copyBtn = document.getElementById('copyBtn');
const speakBtn = document.getElementById('speakBtn');
const autoSpeakToggle = document.getElementById('autoSpeakToggle');
const locBtn = document.getElementById('locBtn');
const locResult = document.getElementById('locResult');
const captureCanvas = document.getElementById('captureCanvas');
const smsBtn112 = document.getElementById('smsBtn112');
const smsBtn119 = document.getElementById('smsBtn119');

let lastLocationLine = '';

function setStatus(active) {
    statusBadge.textContent = active ? '인식 중' : '대기 중';
    statusBadge.classList.toggle('live', active);
    cameraStage.classList.toggle('active', active);
}

function updateSmsLinks() {
    const body = lastLocationLine ? `${reportText.value}\n${lastLocationLine}` : reportText.value;
    const encoded = encodeURIComponent(body);
    smsBtn112.setAttribute('href', `sms:112?body=${encoded}`);
    smsBtn119.setAttribute('href', `sms:119?body=${encoded}`);
}

function speakText(text) {
    if (!text || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    window.speechSynthesis.speak(utterance);
}

function appendReportSentence(sentence) {
    if (!sentence) return;
    const current = reportText.value.trim();
    reportText.value = current ? `${current} ${sentence}` : sentence;
    updateSmsLinks();
    if (autoSpeakToggle.checked) speakText(sentence);
}

socket.on('sign_progress', (data) => {
    if (!data) return;
    if (!data.text && !data.current_jamo) {
        livePreview.textContent = '인식 대기 중...';
        return;
    }
    livePreview.textContent = `인식: ${data.current_jamo || '-'} | 조합: ${data.text || ''}`;
});

socket.on('sign_result', (data) => {
    if (!data || !data.text) return;
    appendReportSentence(data.text);
    livePreview.textContent = '인식 대기 중...';
});

async function startCamera() {
    try {
        myStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        myFace.srcObject = myStream;
        camOffHint.style.display = 'none';
        return true;
    } catch (e) {
        console.error('카메라를 불러올 수 없습니다.', e);
        camOffHint.innerHTML = '<i class="bi bi-camera-video-off"></i>카메라 연결에 실패했어요.<br>브라우저 카메라 권한을 확인해주세요.';
        return false;
    }
}

function captureAndSendFrame() {
    if (!myStream) return;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(myFace, 0, 0, captureCanvas.width, captureCanvas.height);
    const image = captureCanvas.toDataURL('image/jpeg', 0.7);
    socket.emit('sign_frame', { room: ROOM_NAME, name: USER_NAME, image });
}

async function startSignRecognition() {
    if (!myStream) {
        const ok = await startCamera();
        if (!ok) return;
    }
    if (!joined) {
        socket.emit('join_room', { room: ROOM_NAME, name: USER_NAME });
        joined = true;
    }

    signOn = true;
    setStatus(true);
    toggleBtn.classList.add('on');
    toggleBtn.innerHTML = '<i class="bi bi-stop-circle"></i> 수어 인식 중지';
    livePreview.textContent = '인식 대기 중...';
    captureInterval = setInterval(captureAndSendFrame, 100);
}

function stopSignRecognition() {
    signOn = false;
    setStatus(false);
    toggleBtn.classList.remove('on');
    toggleBtn.innerHTML = '<i class="bi bi-camera-video"></i> 수어 인식 시작';
    clearInterval(captureInterval);
}

function toggleSignRecognition() {
    if (signOn) {
        stopSignRecognition();
    } else {
        startSignRecognition();
    }
}

function handleCopy() {
    const text = lastLocationLine ? `${reportText.value}\n${lastLocationLine}` : reportText.value;
    if (!text.trim()) return;
    navigator.clipboard.writeText(text).then(() => {
        const original = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="bi bi-check2"></i> 복사됨';
        setTimeout(() => { copyBtn.innerHTML = original; }, 1500);
    }).catch(() => {});
}

function handleSpeak() {
    speakText(reportText.value.trim());
}

function startLiveLocation() {
    if (!navigator.geolocation) {
        locResult.textContent = '이 브라우저는 위치 정보를 지원하지 않아요.';
        return;
    }
    locResult.textContent = '위치 확인 중...';
    locationWatchId = navigator.geolocation.watchPosition(
        (position) => {
            const { latitude, longitude } = position.coords;
            const lat = latitude.toFixed(5);
            const lng = longitude.toFixed(5);
            const mapUrl = `https://maps.google.com/?q=${lat},${lng}`;
            lastLocationLine = `현재 위치(실시간): ${lat}, ${lng} (${mapUrl})`;
            locResult.innerHTML = `${lat}, ${lng} · <a href="${mapUrl}" target="_blank" rel="noopener">지도에서 보기</a>`;
            updateSmsLinks();
        },
        () => {
            locResult.textContent = '위치 정보를 가져오지 못했어요. 브라우저 권한을 확인해주세요.';
        },
        { enableHighAccuracy: true, timeout: 8000, maximumAge: 2000 }
    );
    locBtn.classList.add('live');
    locBtn.innerHTML = '<i class="bi bi-geo-alt-fill"></i> 실시간 위치 공유 중지';
}

function stopLiveLocation() {
    if (locationWatchId !== null) {
        navigator.geolocation.clearWatch(locationWatchId);
        locationWatchId = null;
    }
    locBtn.classList.remove('live');
    locBtn.innerHTML = '<i class="bi bi-geo-alt"></i> 실시간 위치 공유 시작';
}

function toggleLiveLocation() {
    if (locationWatchId !== null) {
        stopLiveLocation();
    } else {
        startLiveLocation();
    }
}

window.addEventListener('beforeunload', () => {
    if (myStream) myStream.getTracks().forEach((track) => track.stop());
    if (locationWatchId !== null) navigator.geolocation.clearWatch(locationWatchId);
});

document.addEventListener('DOMContentLoaded', () => {
    toggleBtn.addEventListener('click', toggleSignRecognition);
    copyBtn.addEventListener('click', handleCopy);
    speakBtn.addEventListener('click', handleSpeak);
    locBtn.addEventListener('click', toggleLiveLocation);
    reportText.addEventListener('input', updateSmsLinks);
    updateSmsLinks();
});
