// 실시간 텍스트 → 지문자(수어) 번역기
// 입력한 문장을 한글 음절 단위로 초성/중성/종성 자모로 분해한 뒤,
// 순서대로 카드로 재생하여 보여줍니다.

// 실제 손모양 이미지가 준비되면 아래 매핑에 자모 글자를 key로,
// 이미지 경로를 value로 채우면 화면 구성을 바꾸지 않고 바로 반영됩니다.
// 예) 'ㄱ': '/static/images/sign/cho_giyeok.png'
const JAMO_VISUALS = {};

// 사람이 실제로 수어하는 영상이 준비되면, 입력 문장(공백 정리 후 그대로) 또는
// 단어 하나를 key로, 영상 경로를 value로 채워주세요. 등록된 문장/단어가 입력되면
// 지문자 카드 대신 이 영상이 자동으로 재생됩니다.
// 예) '안녕하세요': '/static/videos/sign/annyeonghaseyo.mp4'
const SIGN_VIDEO_MAP = {};

const CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
const JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
const JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
const ROLE_LABEL = { cho: '초성', jung: '중성', jong: '종성', literal: '문자', space: '띄어쓰기' };

const sourceText = document.getElementById('sourceText');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const signStage = document.getElementById('signStage');
const emptyHint = document.getElementById('emptyHint');
const signStrip = document.getElementById('signStrip');
const signVideo = document.getElementById('signVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const playBtn = document.getElementById('playBtn');
const restartBtn = document.getElementById('restartBtn');
const speedSelect = document.getElementById('speedSelect');

let sequence = [];
let currentIndex = -1;
let playing = false;
let playTimer = null;
let debounceTimer = null;

function splitSyllable(char) {
    const code = char.codePointAt(0) - 0xac00;
    if (code < 0 || code > 11171) return null;
    const choIdx = Math.floor(code / 588);
    const jungIdx = Math.floor((code % 588) / 28);
    const jongIdx = code % 28;
    return {
        cho: CHO[choIdx],
        jung: JUNG[jungIdx],
        jong: jongIdx === 0 ? null : JONG[jongIdx],
    };
}

function buildSequence(text) {
    const units = [];
    for (const char of Array.from(text)) {
        if (/\s/.test(char)) {
            units.push({ type: 'space' });
            continue;
        }
        const parts = splitSyllable(char);
        if (parts) {
            units.push({ type: 'cho', char: parts.cho, syllable: char });
            units.push({ type: 'jung', char: parts.jung, syllable: char });
            if (parts.jong) units.push({ type: 'jong', char: parts.jong, syllable: char });
        } else {
            units.push({ type: 'literal', char });
        }
    }
    return units;
}

function getVisual(char) {
    return JAMO_VISUALS[char] || null;
}

function findSignVideo(text) {
    const trimmed = text.trim();
    if (!trimmed) return null;
    if (SIGN_VIDEO_MAP[trimmed]) return SIGN_VIDEO_MAP[trimmed];
    const firstWord = trimmed.split(/\s+/)[0];
    return SIGN_VIDEO_MAP[firstWord] || null;
}

function updateVideo(text) {
    const src = findSignVideo(text);
    if (src) {
        if (signVideo.getAttribute('src') !== src) {
            signVideo.setAttribute('src', src);
        }
        signVideo.classList.remove('hidden');
        videoPlaceholder.classList.add('hidden');
        signVideo.currentTime = 0;
        signVideo.play().catch(() => {});
    } else {
        signVideo.pause();
        signVideo.removeAttribute('src');
        signVideo.load();
        signVideo.classList.add('hidden');
        videoPlaceholder.classList.remove('hidden');
    }
}

function renderStrip() {
    signStrip.innerHTML = '';
    sequence.forEach((unit, index) => {
        const card = document.createElement('div');
        if (unit.type === 'space') {
            card.className = 'strip-card space';
        } else {
            card.className = 'strip-card';
            card.textContent = unit.char;
            card.dataset.index = String(index);
        }
        signStrip.appendChild(card);
    });
}

function updateStripHighlight() {
    const cards = signStrip.querySelectorAll('.strip-card[data-index]');
    cards.forEach((card) => {
        const index = Number(card.dataset.index);
        card.classList.toggle('active', index === currentIndex);
        card.classList.toggle('done', index < currentIndex);
    });
    const activeCard = signStrip.querySelector('.strip-card.active');
    if (activeCard) activeCard.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}

function renderStage(unit) {
    signStage.innerHTML = '';

    if (!unit) {
        signStage.appendChild(emptyHint);
        return;
    }

    const card = document.createElement('div');
    card.className = 'stage-card pulse';

    const visual = unit.type !== 'space' ? getVisual(unit.char) : null;
    if (visual) {
        const img = document.createElement('img');
        img.src = visual;
        img.alt = unit.char || '띄어쓰기';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        card.appendChild(img);
    } else {
        card.textContent = unit.type === 'space' ? '␣' : unit.char;
    }

    const meta = document.createElement('div');
    meta.className = 'stage-meta';
    const syllableEl = document.createElement('div');
    syllableEl.className = 'syllable';
    syllableEl.textContent = unit.syllable || (unit.type === 'space' ? '(공백)' : unit.char);
    const roleEl = document.createElement('div');
    roleEl.className = 'role';
    roleEl.textContent = ROLE_LABEL[unit.type] || '';
    meta.appendChild(syllableEl);
    meta.appendChild(roleEl);

    signStage.appendChild(card);
    signStage.appendChild(meta);

    requestAnimationFrame(() => card.classList.remove('pulse'));
}

function setPlayIcon(isPlaying) {
    playBtn.innerHTML = isPlaying ? '<i class="bi bi-pause-fill"></i>' : '<i class="bi bi-play-fill"></i>';
}

function stopPlayback() {
    playing = false;
    clearTimeout(playTimer);
    setPlayIcon(false);
}

function tick() {
    currentIndex += 1;
    if (currentIndex >= sequence.length) {
        stopPlayback();
        return;
    }
    renderStage(sequence[currentIndex]);
    updateStripHighlight();
    const speed = Number(speedSelect.value) || 550;
    playTimer = setTimeout(tick, speed);
}

function startPlayback() {
    if (sequence.length === 0) return;
    if (currentIndex >= sequence.length - 1) currentIndex = -1;
    playing = true;
    setPlayIcon(true);
    tick();
}

function togglePlayback() {
    if (playing) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function restartPlayback() {
    stopPlayback();
    currentIndex = -1;
    updateStripHighlight();
    startPlayback();
}

function resetView() {
    sequence = [];
    currentIndex = -1;
    stopPlayback();
    signStrip.innerHTML = '';
    renderStage(null);
    updateVideo('');
}

function handleInputChange() {
    const text = sourceText.value;
    charCount.textContent = String(text.length);

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        stopPlayback();
        const trimmed = text.trim();
        if (!trimmed) {
            resetView();
            return;
        }
        updateVideo(text);
        sequence = buildSequence(text);
        currentIndex = -1;
        renderStrip();
        startPlayback();
    }, 250);
}

function handleClear() {
    sourceText.value = '';
    charCount.textContent = '0';
    resetView();
    sourceText.focus();
}

document.addEventListener('DOMContentLoaded', () => {
    renderStage(null);
    sourceText.addEventListener('input', handleInputChange);
    clearBtn.addEventListener('click', handleClear);
    playBtn.addEventListener('click', togglePlayback);
    restartBtn.addEventListener('click', restartPlayback);
});
