let roomStream;

$(document).ready(function() {
    const roomDataStr = localStorage.getItem('roomData');
    if(!roomDataStr) {
        alert("비정상적인 접근입니다.");
        window.location.href = '/chatentry.html';
        return;
    }

    const roomData = JSON.parse(roomDataStr);
    $('#dynamicTitle').text(`[방 번호: ${roomData.room}] ${roomData.name}님`);

    // 1. 카메라 연동 및 초기 설정 (입장 전 설정값 반영)
    async function startRoomCamera() {
        try {
            roomStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            const videoEl = document.getElementById('myFace');
            videoEl.srcObject = roomStream;
            videoEl.muted = true; // 본인 마이크 하울링 방지

            // 입장 전 설정한 오디오/비디오 상태 적용
            if(roomData.isAudioOn === false) $('#myAudioStat').click();
            if(roomData.isVideoOn === false) $('#myVideoStat').click();
        } catch (e) {
            console.error("카메라를 불러올 수 없습니다.", e);
            $('#subtitleText').text("카메라 연결 실패");
        }
    }
    startRoomCamera();

    // 2. 하단 컨트롤러 토글 로직
    $('#myAudioStat').click(function() {
        if(roomStream) {
            const track = roomStream.getAudioTracks()[0];
            track.enabled = !track.enabled;
            $(this).toggleClass('active off');
            $(this).find('i').attr('class', track.enabled ? 'bi bi-mic-fill' : 'bi bi-mic-mute-fill');
            $(this).find('span').text(track.enabled ? '음소거' : '음소거 해제');
        }
    });

    $('#myVideoStat').click(function() {
        if(roomStream) {
            const track = roomStream.getVideoTracks()[0];
            track.enabled = !track.enabled;
            $(this).toggleClass('active off');
            $(this).find('i').attr('class', track.enabled ? 'bi bi-camera-video-fill' : 'bi bi-camera-video-off-fill');
            $(this).find('span').text(track.enabled ? '비디오' : '비디오 켜기');
        }
    });

    $('#RC_sign_toggle').click(function() {
        $(this).toggleClass('active off');
        const isOn = $(this).hasClass('active');
        $(this).find('span').text(isOn ? '수어 인식 ON' : '수어 인식 OFF');
        $('#subtitleText').text(isOn ? 'AI 수어 인식 대기 중...' : '수어 인식 기능이 꺼졌습니다.');
    });

    // 3. 실시간 채팅 UI 연동
    function appendMyMessage(text) {
        const msgHtml = `<div class="chat-bubble me">${text.replace(/\n/g, '<br>')}</div>`;
        $('#chatBox').append(msgHtml);
        $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);
    }

    $('#chatForm').on('submit', function(e) {
        e.preventDefault(); 
        const msg = $('#userChat').val().trim();
        if (msg !== '') {
            appendMyMessage(msg);
            $('#userChat').val('');
        }
    });

    $('#userChat').on('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            $('#chatForm').trigger('submit');
        }
    });

    // 4. 파일 업로드 로직
    $('#fileUpload').on('change', function(e) {
        const file = e.target.files[0];
        if(file) {
            const fileHtml = `
                <div class="chat-bubble me" style="display:flex; align-items:center; gap:8px;">
                    <i class="bi bi-file-earmark-text-fill" style="font-size:24px;"></i>
                    <div>
                        <div style="font-weight:700;">${file.name}</div>
                        <div style="font-size:11px; opacity:0.8;">전송 완료</div>
                    </div>
                </div>`;
            $('#chatBox').append(fileHtml);
            $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);
            $(this).val(''); // 초기화
        }
    });
});

function resetPage() {
    localStorage.removeItem('roomData');
    if(roomStream) {
        roomStream.getTracks().forEach(track => track.stop());
    }
    window.location.href = '/chatentry.html';
}