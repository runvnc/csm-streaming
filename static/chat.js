let ws;
let sessionStartTime = null;
let messageCount = 0;
let audioLevelsChart = null;
let isRecording = false;
let isAudioCurrentlyPlaying = false;
let configSaved = false;
let currentAudioSource = null; 
let interruptRequested = false; 
let audioContext = null;
let reconnecting = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 10;

const SESSION_ID = "default";
console.log("chat.js loaded - optimized for low latency");

let micStream;
let selectedMicId = null;
let selectedOutputId = null;

let audioPlaybackQueue = [];
let micAnalyser, micContext;
let activeGenId = 0;

function createPermanentVoiceCircle() {
  if (document.getElementById('voice-circle')) return;
  const style = document.createElement('style');
  style.textContent = `
    #voice-circle{
      position:fixed;top:50%;left:50%;
      width:180px;height:180px;border-radius:50%;
      background:rgba(99,102,241,.20);
      transform:translate(-50%,-50%) scale(var(--dynamic-scale,1));
      pointer-events:none;z-index:50;
      transition:background-color .35s ease;
    }
    #voice-circle.active{
      animation:pulse-circle 2s infinite alternate ease-in-out;
    }
    @keyframes pulse-circle{
      0%{background:rgba(99,102,241,.55)}
      100%{background:rgba(99,102,241,.20)}
    }`;
  document.head.appendChild(style);

  const c = document.createElement('div');
  c.id='voice-circle';
  document.body.appendChild(c);
  console.log("Created permanent voice circle");
}

function showVoiceCircle() {
  const c=document.getElementById('voice-circle')||createPermanentVoiceCircle();
  if (c) c.classList.add('active');
}

function hideVoiceCircle() {
  const c=document.getElementById('voice-circle');
  if (c){ c.classList.remove('active'); c.style.setProperty('--dynamic-scale',1); }
}

function showNotification(msg, type='info'){
  const n=document.createElement('div');
  n.className=`fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg z-50
               ${type==='success'?'bg-green-600':
                 type==='error'  ?'bg-red-600':'bg-indigo-600'}`;
  n.textContent=msg;
  document.body.appendChild(n);
  setTimeout(()=>{n.classList.add('opacity-0');
                  setTimeout(()=>n.remove(),500)},3000);
}

function addMessageToConversation(sender,text){
  const pane=document.getElementById('conversationHistory');
  if(!pane) return;
  const box=document.createElement('div');
  box.className=`p-3 mb-3 rounded-lg text-sm ${
            sender==='user'?'bg-gray-800 ml-2':'bg-indigo-900 mr-2'}`;
  box.innerHTML=`
      <div class="flex items-start mb-2">
        <div class="w-6 h-6 rounded-full flex items-center justify-center
             ${sender==='user'?'bg-gray-300 text-gray-800':'bg-indigo-500 text-white'}">
             ${sender==='user'?'U':'AI'}
        </div>
        <span class="text-xs text-gray-400 ml-2">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="text-white mt-1 text-sm">${text
            .replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
            .replace(/\*(.*?)\*/g,'<em>$1</em>')
            .replace(/```([^`]+)```/g,'<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g,'<code>$1</code>')
            .replace(/\n/g,'<br>')}</div>`;
  pane.appendChild(box);
  pane.scrollTop=pane.scrollHeight;
}

function connectWebSocket() {
  if (reconnecting && reconnectAttempts >= maxReconnectAttempts) {
    console.error("Maximum reconnect attempts reached. Please refresh the page.");
    showNotification("Connection lost. Please refresh the page.", "error");
    return;
  }

  if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
    try {
      ws.close();
    } catch (e) {
      console.warn("Error closing existing WebSocket:", e);
    }
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  window.ws = ws;

  const connLbl = document.getElementById('connectionStatus');
  if (connLbl) {
    connLbl.textContent = reconnecting ? 'Reconnecting…' : 'Connecting…';
    connLbl.className = 'text-yellow-500';
  }

  ws.onopen = () => {
    if (connLbl) {
      connLbl.textContent = 'Connected';
      connLbl.className = 'text-green-500';
    }
    
    reconnecting = false;
    reconnectAttempts = 0;
    
    ws.send(JSON.stringify({type: 'request_saved_config'}));
    
    if (!reconnecting) {
      addMessageToConversation('ai', 'WebSocket connected. Ready for voice or text.');
    } else {
      showNotification("Reconnected successfully", "success");
    }
  };

  ws.onclose = (event) => {
    console.log("WebSocket closed with code:", event.code);
    if (connLbl) {
      connLbl.textContent = 'Disconnected';
      connLbl.className = 'text-red-500';
    }

    clearAudioPlayback();
    
    if (event.code !== 1000 && event.code !== 1001) {
      reconnecting = true;
      reconnectAttempts++;
      
      const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 1000);
      console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
      
      setTimeout(connectWebSocket, delay);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    if (connLbl) {
      connLbl.textContent = 'Error';
      connLbl.className = 'text-red-500';
    }
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleWebSocketMessage(data);
    } catch (err) {
      console.error("Error handling WebSocket message:", err);
    }
  };
}

function sendTextMessage(txt) {
  if (!txt.trim()) return;
  
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showNotification("Not connected", "error");
    return;
  }
  
  // Stop any playing audio immediately
  if (isAudioCurrentlyPlaying) {
    clearAudioPlayback();
  }
  
  // Reset state for new interaction
  interruptRequested = false;
  activeGenId = 0;
  audioPlaybackQueue = [];
  
  // Send interrupt to server
  try {
    ws.send(JSON.stringify({type: 'interrupt', immediate: true}));
  } catch (e) {
    console.warn("Error sending interrupt:", e);
  }
  
  // Send message immediately (no delay needed)
  try {
    showVoiceCircle();
    
    ws.send(JSON.stringify({
      type: 'text_message',
      text: txt,
      session_id: SESSION_ID
    }));
    
    const cnt = document.getElementById('messageCount');
    if (cnt) cnt.textContent = ++messageCount;
    
    document.getElementById('textInput').value = '';
    
    console.log("Text message sent");
  } catch (error) {
    console.error("Error sending message:", error);
    showNotification("Error sending message", "error");
  }
}

function clearAudioPlayback() {
  console.log("Clearing audio playback");
  
  interruptRequested = true;
  audioPlaybackQueue = [];
  activeGenId = 0;
  
  // Stop current audio source
  if (currentAudioSource) {
    try {
      currentAudioSource.disconnect();
      currentAudioSource.stop(0);
    } catch (e) {
      // Ignore errors - source may already be stopped
    }
    currentAudioSource = null;
  }
  
  isAudioCurrentlyPlaying = false;
  hideVoiceCircle();
  
  // Reset interrupt flag after a brief moment
  setTimeout(() => {
    interruptRequested = false;
  }, 100);
}

function requestInterrupt() {
  console.log("User requested interruption");
  
  clearAudioPlayback();
  
  // Notify server
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify({type: 'interrupt', immediate: true}));
    } catch (error) {
      console.error("Error sending interrupt request:", error);
    }
  }
  
  return true;
}

function handleWebSocketMessage(d) {
  // Only log non-audio messages to reduce console spam
  if (d.type !== 'audio_chunk') {
    console.log("Received:", d.type, d.status || '');
  }
  
  switch(d.type) {
    case 'transcription':
      addMessageToConversation('user', d.text);
      showVoiceCircle();
      break;
      
    case 'response':
      addMessageToConversation('ai', d.text);
      showVoiceCircle();
      
      // Reset state for new response
      interruptRequested = false;
      activeGenId = 0;
      audioPlaybackQueue = [];
      
      // Ensure audio context is ready
      ensureAudioContext();
      break;
      
    case 'audio_chunk':
      // Accept audio if not interrupted and matches current generation
      if (interruptRequested) {
        return;
      }
      
      const chunkGenId = d.gen_id || 1;
      
      // First chunk of a new generation
      if (activeGenId === 0) {
        activeGenId = chunkGenId;
        console.log("Starting new audio generation:", activeGenId);
      }
      
      // Only accept chunks from current generation
      if (chunkGenId === activeGenId) {
        queueAudioForPlayback(d.audio, d.sample_rate, chunkGenId);
        showVoiceCircle();
      }
      break;
      
    case 'audio_status':
      if (d.status === 'generating') {
        // New generation starting - reset state
        interruptRequested = false;
        if (d.gen_id) {
          activeGenId = d.gen_id;
        }
        showVoiceCircle();
        ensureAudioContext();
      } 
      else if (d.status === 'complete') {
        if (!d.gen_id || d.gen_id === activeGenId) {
          activeGenId = 0;
        }
        if (!isAudioCurrentlyPlaying && audioPlaybackQueue.length === 0) {
          hideVoiceCircle();
        }
      } 
      else if (d.status === 'interrupted') {
        clearAudioPlayback();
      }
      break;
      
    case 'status':
      if (d.message === 'Thinking...') {
        showVoiceCircle();
        interruptRequested = false;
        activeGenId = 0;
      }
      break;
      
    case 'error':
      showNotification(d.message, 'error');
      hideVoiceCircle();
      break;
      
    case 'vad_status':
      if (d.status === 'speech_started' && d.should_interrupt && isAudioCurrentlyPlaying) {
        requestInterrupt();
      }
      break;
  }
}

function ensureAudioContext() {
  if (!audioContext || audioContext.state === 'closed') {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    window.audioContext = audioContext;
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume().catch(e => console.warn("Error resuming audio context:", e));
  }
}

function queueAudioForPlayback(arr, sr, genId) {
  if (interruptRequested) {
    return;
  }
  
  if (activeGenId !== 0 && genId !== activeGenId) {
    console.log("Ignoring stale audio chunk");
    return;
  }
  
  audioPlaybackQueue.push({arr, sr, genId});
  
  if (!isAudioCurrentlyPlaying) {
    processAudioPlaybackQueue();
  }
}

function processAudioPlaybackQueue() {
  if (interruptRequested || audioPlaybackQueue.length === 0) {
    isAudioCurrentlyPlaying = false;
    if (audioPlaybackQueue.length === 0) {
      hideVoiceCircle();
    }
    return;
  }
  
  isAudioCurrentlyPlaying = true;
  const {arr, sr, genId} = audioPlaybackQueue.shift();
  
  // Skip stale chunks
  if (activeGenId !== 0 && genId !== activeGenId) {
    processAudioPlaybackQueue();
    return;
  }
  
  playAudioChunk(arr, sr)
    .then(() => {
      if (!interruptRequested) {
        processAudioPlaybackQueue();
      } else {
        isAudioCurrentlyPlaying = false;
        hideVoiceCircle();
      }
    })
    .catch(err => {
      console.error("Audio playback error:", err);
      isAudioCurrentlyPlaying = false;
      // Try next chunk
      if (audioPlaybackQueue.length > 0 && !interruptRequested) {
        setTimeout(processAudioPlaybackQueue, 50);
      }
    });
}

async function playAudioChunk(audioArr, sampleRate) {
  if (interruptRequested) {
    return;
  }
  
  ensureAudioContext();
  
  const buf = audioContext.createBuffer(1, audioArr.length, sampleRate);
  buf.copyToChannel(new Float32Array(audioArr), 0);
  
  const src = audioContext.createBufferSource();
  src.buffer = buf;
  currentAudioSource = src;
  
  const analyser = audioContext.createAnalyser(); 
  analyser.fftSize = 256;
  src.connect(analyser); 
  analyser.connect(audioContext.destination); 
  src.start();

  // Animate voice circle based on audio levels
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  const circle = document.getElementById('voice-circle');
  
  function animate() {
    if (src !== currentAudioSource || interruptRequested) return;
    
    analyser.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a,b) => a+b, 0) / dataArray.length;
    if (circle) {
      circle.style.setProperty('--dynamic-scale', (1 + avg/255 * 1.5).toFixed(3));
    }
    requestAnimationFrame(animate);
  }
  animate();
  
  return new Promise(resolve => {
    src.onended = resolve;
  });
}

async function startRecording() {
  if (isRecording) return;
  try {
    const constraints = {
      audio: selectedMicId ? {deviceId:{exact:selectedMicId}} : true
    };
    micStream = await navigator.mediaDevices.getUserMedia(constraints);

    if (!audioContext) audioContext = new (AudioContext||webkitAudioContext)();
    const src = audioContext.createMediaStreamSource(micStream);
    const proc = audioContext.createScriptProcessor(4096,1,1);
    src.connect(proc); proc.connect(audioContext.destination);

    proc.onaudioprocess = e => {
      const samples = Array.from(e.inputBuffer.getChannelData(0));
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({
            type:'audio',
            audio:samples,
            sample_rate:audioContext.sampleRate,
            session_id:SESSION_ID
          }));
        } catch (error) {
          console.error("Error sending audio data:", error);
          stopRecording();
        }
      }
    };

    window._micProcessor = proc;        
    isRecording = true;
    const micStatus = document.getElementById('micStatus');
    if (micStatus) micStatus.textContent = 'Listening…';
    showVoiceCircle();
  } catch (err) {
    console.error("Microphone access error:", err);
    showNotification('Microphone access denied','error');
  }
}

function stopRecording() {
  if (!isRecording) return;
  try {
    if (window._micProcessor) {
      window._micProcessor.disconnect();
      window._micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
  } catch(e) {
    console.warn("Error stopping recording:", e);
  }
  isRecording = false;
  
  const micStatus = document.getElementById('micStatus');
  if (micStatus) micStatus.textContent = 'Click to speak';
  hideVoiceCircle();
}

async function setupChatUI() {
  document.documentElement.classList.add('bg-gray-950');
  document.documentElement.style.backgroundColor = '#030712';

  createPermanentVoiceCircle();
  connectWebSocket();
  initAudioLevelsChart();

  const txt = document.getElementById('textInput');
  const btn = document.getElementById('sendTextBtn');
  
  // Interrupt button
  const interruptBtn = document.createElement('button');
  interruptBtn.id = 'interruptBtn';
  interruptBtn.className = 'px-3 py-2 ml-2 bg-red-600 text-white rounded hover:bg-red-700 flex items-center transition duration-150';
  interruptBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd" /></svg> Stop';
  interruptBtn.onclick = (e) => {
    e.preventDefault();
    requestInterrupt();
  };
  interruptBtn.title = "Stop AI speech (Space or Esc)";
  
  if (btn && btn.parentElement) {
    btn.parentElement.appendChild(interruptBtn);
  }
  
  if (btn) {
    btn.onclick = () => sendTextMessage(txt.value);
  }
  
  if (txt) {
    txt.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendTextMessage(txt.value);
      }
    });
  }
  
  const micBtn = document.getElementById('micToggleBtn');
  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (isRecording) stopRecording();
      else startRecording();
    });
  }
  
  // Keyboard shortcuts for interrupt
  document.addEventListener('keydown', e => {
    if ((e.code === 'Space' || e.code === 'Escape') && isAudioCurrentlyPlaying) {
      e.preventDefault();
      requestInterrupt();
    }
  });
  
  // Initialize audio context on first user interaction
  ['click', 'touchstart', 'keydown'].forEach(ev =>
    document.addEventListener(ev, function unlock() {
      ensureAudioContext();
      document.removeEventListener(ev, unlock);
    }, {once: true})
  );

  console.log("Chat UI ready - optimized for low latency");
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupChatUI);
} else {
  setupChatUI();
}

function initAudioLevelsChart() {
  const ctx = document.getElementById('audioLevels');
  if (!ctx) return;
  
  try {
    if (audioLevelsChart) audioLevelsChart.destroy();
    
    const grad = ctx.getContext('2d').createLinearGradient(0, 0, 0, 100);
    grad.addColorStop(0, 'rgba(79,70,229,.6)');
    grad.addColorStop(1, 'rgba(79,70,229,.1)');
    
    audioLevelsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [{
          data: Array(30).fill(0),
          backgroundColor: grad,
          borderColor: 'rgba(99,102,241,1)',
          borderWidth: 2,
          tension: .4,
          fill: true,
          pointRadius: 0
        }]
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {display: false},
            grid: {color: 'rgba(255,255,255,.1)'}
          },
          x: {display: false, grid: {display: false}}
        },
        plugins: {
          legend: {display: false},
          tooltip: {enabled: false}
        },
        elements: {point: {radius: 0}}
      }
    });
  } catch (error) {
    console.error("Error initializing audio chart:", error);
  }
}
