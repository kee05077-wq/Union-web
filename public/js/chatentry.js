let localStream;
let isAudioOn = true;
let isVideoOn = true;

// 카메라 및 마이크 초기화
async function initPreview() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        const video = document.createElement('video');
        video.srcObject = localStream;
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true; // 본인 하울링 방지
        
        document.getElementById('preview').innerHTML = '';
        document.getElementById('preview').appendChild(video);
    } catch (err) {
        document.getElementById('preview').innerText = "카메라/마이크 권한을 허용해주세요.";
    }
}

function toggleAudio() {
    if(!localStream) return;
    isAudioOn = !isAudioOn;
    localStream.getAudioTracks()[0].enabled = isAudioOn;
    const btn = document.getElementById('btn-audio');
    btn.className = isAudioOn ? '' : 'off';
    btn.innerHTML = isAudioOn ? '<i class="bi bi-mic-fill"></i>' : '<i class="bi bi-mic-mute-fill"></i>';
}

function toggleVideo() {
    if(!localStream) return;
    isVideoOn = !isVideoOn;
    localStream.getVideoTracks()[0].enabled = isVideoOn;
    const btn = document.getElementById('btn-video');
    btn.className = isVideoOn ? '' : 'off';
    btn.innerHTML = isVideoOn ? '<i class="bi bi-camera-video-fill"></i>' : '<i class="bi bi-camera-video-off-fill"></i>';
}

function joinRoom() {
    const room = document.getElementById('roomNumber').value.trim();
    const name = document.getElementById('userName').value.trim();
    if(room && name) {
        localStorage.setItem('roomData', JSON.stringify({room, name, isAudioOn, isVideoOn}));
        // 경로 변경됨
        window.location.href = '/room.html';
    } else {
        alert("방 번호와 닉네임을 모두 입력해주세요.");
    }
}

$(document).ready(function() {
    initPreview();
});