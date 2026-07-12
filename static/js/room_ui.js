let roomStream;
let peerConnection;
let captureInterval;

let isMySignOn = true;    
let isViewSignOn = true;  

const socket = io();
const iceServers = {
    'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }]
};

$(document).ready(function() {
    const roomDataStr = localStorage.getItem('roomData');
    if(!roomDataStr) {
        alert("비정상적인 접근입니다.");
        window.location.href = '/chatentry.html';
        return;
    }

    const roomData = JSON.parse(roomDataStr);
    $('#dynamicTitle').text(`[방 번호: ${roomData.room}] ${roomData.name}님`);

    async function startRoomCamera() {
        try {
            roomStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            
            const myFaceEl = document.getElementById('myFace');
            myFaceEl.srcObject = roomStream;
            myFaceEl.style.transform = "scaleX(-1)"; // 본인 화면은 거울 모드로 표시 (오른손이 화면 오른쪽에 보이도록)

            if(roomData.isAudioOn === false) $('#myAudioStat').click();
            if(roomData.isVideoOn === false) $('#myVideoStat').click();
            
            socket.emit('join_room', { room: roomData.room, name: roomData.name });
            startSignCapture();
        } catch (e) {
            console.error("카메라 연결 실패", e);
        }
    }
    startRoomCamera();

    function createPeerConnection() {
        peerConnection = new RTCPeerConnection(iceServers);
        roomStream.getTracks().forEach(track => peerConnection.addTrack(track, roomStream));

        peerConnection.ontrack = (event) => {
            const remoteVideo = document.getElementById('remoteFace');
            if (remoteVideo.srcObject !== event.streams[0]) {
                remoteVideo.srcObject = event.streams[0];
            }
        };

        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                socket.emit('webrtc_ice', { room: roomData.room, candidate: event.candidate });
            }
        };
    }

    socket.on('user_joined', async (data) => {
        createPeerConnection();
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        socket.emit('webrtc_offer', { room: roomData.room, sdp: offer });
    });

    socket.on('webrtc_offer', async (data) => {
        createPeerConnection();
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        socket.emit('webrtc_answer', { room: roomData.room, sdp: answer });
    });

    socket.on('webrtc_answer', async (data) => {
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
    });

    socket.on('webrtc_ice', async (data) => {
        if (peerConnection) {
            await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
        }
    });

    function startSignCapture() {
        const videoEl = document.getElementById('myFace');
        const canvas = document.getElementById('hiddenCanvas');
        const context = canvas.getContext('2d');

        captureInterval = setInterval(() => {
            if (isMySignOn && roomStream.getVideoTracks()[0].enabled) {
                context.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
                const frameData = canvas.toDataURL('image/jpeg', 0.7);
                socket.emit('sign_frame', { room: roomData.room, name: roomData.name, image: frameData });
            }
        }, 29); // 초당 약 35프레임 (1000ms / 35 ≈ 29ms)
    }

    // 듀얼 피드백 적용: 인식된 자모와 조합 텍스트 표출
    socket.on('sign_progress', function(data) {
        if (isMySignOn) {
            if (data.text === '' && (data.current_jamo === '' || !data.current_jamo)) {
                $('#mySignPreview').hide();
            } else {
                const currentJamoHtml = `<span style="color:#ffeb3b;">인식: ${data.current_jamo || '-'}</span>`;
                const assembledHtml = `<span>조합: ${data.text}</span>`;
                
                $('#mySignPreview').show().html(`${currentJamoHtml} &nbsp;|&nbsp; ${assembledHtml}`);
            }
        }
    });

    socket.on('sign_result', function(data) {
        if (data.name === roomData.name) {
            $('#mySignPreview').hide().text('');
        }

        if (isViewSignOn) {
            $('#subtitleText').show().text(`[${data.name}] ${data.text}`);
            clearTimeout(window.subtitleTimer);
            window.subtitleTimer = setTimeout(() => {
                $('#subtitleText').fadeOut();
            }, 4000);
        }
    });

    $('#toggleMySign').click(function() {
        $(this).toggleClass('active off');
        isMySignOn = $(this).hasClass('active');
        $(this).find('span').text(isMySignOn ? '내 수어 전송 ON' : '내 수어 전송 OFF');
        if(!isMySignOn) $('#mySignPreview').hide();
    });

    $('#toggleViewSign').click(function() {
        $(this).toggleClass('active off');
        isViewSignOn = $(this).hasClass('active');
        $(this).find('span').text(isViewSignOn ? '자막 켜짐' : '자막 꺼짐');
        if(!isViewSignOn) $('#subtitleText').hide();
    });

    $('#chatForm').on('submit', function(e) {
        e.preventDefault(); 
        const msg = $('#userChat').val().trim();
        if (msg !== '') {
            socket.emit('send_message', { room: roomData.room, name: roomData.name, msg: msg });
            $('#userChat').val('');
        }
    });

    socket.on('receive_message', function(data) {
        if($('#chatBox .chat-bubble').length === 1 && $('#chatBox .chat-bubble').css('background-color') === 'transparent') {
            $('#chatBox').html('');
        }
        
        const isMe = (data.name === roomData.name);
        const alignClass = isMe ? 'me' : 'other';
        const nameTag = isMe ? '' : `<div style="font-size:11px; margin-bottom:4px; opacity:0.7;">${data.name}</div>`;
        const msgHtml = `<div class="chat-bubble ${alignClass}">${nameTag}${data.msg.replace(/\n/g, '<br>')}</div>`;
        
        $('#chatBox').append(msgHtml);
        $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);
    });

    $('#myAudioStat').click(function() {
        const track = roomStream.getAudioTracks()[0];
        track.enabled = !track.enabled;
        $(this).toggleClass('active off');
        $(this).find('i').attr('class', track.enabled ? 'bi bi-mic-fill' : 'bi bi-mic-mute-fill');
        $(this).find('span').text(track.enabled ? '음소거' : '음소거 해제');
    });

    $('#myVideoStat').click(function() {
        const track = roomStream.getVideoTracks()[0];
        track.enabled = !track.enabled;
        $(this).toggleClass('active off');
        $(this).find('i').attr('class', track.enabled ? 'bi bi-camera-video-fill' : 'bi bi-camera-video-off-fill');
        $(this).find('span').text(track.enabled ? '비디오' : '비디오 켜기');
    });
});

function resetPage() {
    localStorage.removeItem('roomData');
    if(roomStream) roomStream.getTracks().forEach(t => t.stop());
    if(peerConnection) peerConnection.close();
    window.location.href = '/chatentry.html';
}