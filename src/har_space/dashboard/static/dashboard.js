/**
 * AstroHAR — Astronaut Activity Monitor (Simplified Dashboard Controller)
 */
(function() {
  'use strict';

  const state = {
    steps: [],
    currentStepIdx: 0,
    completedSteps: new Set(),
    stepStatuses: {},
    actionProbs: {},
    voiceText: 'Waiting for an instruction.',
    wrongAction: null,
    logs: []
  };

  const dom = {
    clock: document.getElementById('clock-display'),
    stepBadge: document.getElementById('step-badge'),
    stepTitle: document.getElementById('step-title'),
    actionDetected: document.getElementById('action-detected'),
    targetObject: document.getElementById('target-object'),
    warningBanner: document.getElementById('warning-banner'),
    warningText: document.getElementById('warning-text'),
    stepList: document.getElementById('step-list'),
    voiceText: document.getElementById('voice-text'),
    logList: document.getElementById('log-list'),
    cvCanvas: document.getElementById('cv-canvas'),
    detectionList: document.getElementById('detection-list'),
    feedCaption: document.getElementById('feed-caption'),
    elapsed: document.getElementById('elapsed-val'),
    inputValue: document.getElementById('input-value'),
    protocolStatus: document.getElementById('protocol-status'),
    btnExportLog: document.getElementById('btn-export-log')
  };

  function init() {
    startClock();
    setupListeners();
    renderUI();
    setInterval(pollBackendState, 1000);
  }

  function pollBackendState() {
    fetch('/api/state')
      .then(res => res.json())
      .then(data => {
        if (data) {
          if (data.current_step_idx !== undefined) {
            state.currentStepIdx = data.current_step_idx;
          }
          if (data.fps !== undefined) {
            const fpsEl = document.getElementById('fps-val');
            if (fpsEl) fpsEl.textContent = data.fps;
          }
          if (Array.isArray(data.steps)) state.steps = data.steps;
          if (Array.isArray(data.completed_steps)) state.completedSteps = new Set(data.completed_steps);
          if (data.step_statuses) state.stepStatuses = data.step_statuses;
          if (Array.isArray(data.logs)) state.logs = data.logs;
          if (data.action_probs) state.actionProbs = data.action_probs;
          state.voiceText = data.voice_text || state.voiceText;
          renderLiveState(data);
          renderUI();
        }
      })
      .catch(() => {});
  }

  function startClock() {
    setInterval(() => {
      const now = new Date();
      dom.clock.textContent = now.toTimeString().split(' ')[0] + ' UTC';
    }, 1000);
  }

  function renderUI() {
    if (!state.steps.length) return;
    const complete = state.currentStepIdx >= state.steps.length;
    const cur = state.steps[Math.min(state.currentStepIdx, state.steps.length - 1)];
    dom.stepBadge.textContent = complete ? 'COMPLETE' : `STEP ${state.currentStepIdx + 1} / ${state.steps.length}`;
    dom.stepTitle.textContent = complete ? 'Experiment complete.' : cur.instruction;
    dom.targetObject.textContent = complete ? 'All steps completed' : cur.name.replace('Pick ', '').replace('Place ', '').replace('Locate ', '');
    dom.voiceText.textContent = state.voiceText ? `"${state.voiceText}"` : 'Waiting for an instruction.';

    if (complete) {
      dom.actionDetected.textContent = 'COMPLETE (100%)';
      dom.actionDetected.className = 'metric-val text-green';
      dom.warningBanner.classList.add('hidden');
      renderTimeline();
      renderLogs();
      return;
    }

    if (state.wrongAction) {
      dom.actionDetected.textContent = `${state.wrongAction} (89%)`;
      dom.actionDetected.className = 'metric-val text-amber';
      dom.warningBanner.classList.remove('hidden');
      dom.warningText.textContent = `Expected: ${cur.action} | Detected: ${state.wrongAction}`;
    } else {
      const actionEntries = Object.entries(state.actionProbs || {});
      const latestAction = actionEntries.length ? actionEntries[0][0] : cur.action;
      const latestConfidence = actionEntries.length ? Math.round(actionEntries[0][1] * 100) : 0;
      dom.actionDetected.textContent = `${latestAction} (${latestConfidence}%)`;
      dom.actionDetected.className = 'metric-val text-cyan';
      dom.warningBanner.classList.add('hidden');
    }

    renderTimeline();
    renderLogs();
  }

  function renderLiveState(data) {
    const sourceMode = data.source_mode || 'live';
    dom.inputValue.textContent = sourceMode === 'replay' ? 'Recorded video' : (sourceMode === 'idle' ? 'Stopped' : 'Live camera');
    dom.feedCaption.textContent = sourceMode === 'replay' ? 'Recorded video • AI overlay' : 'Live camera • AI overlay';
    dom.protocolStatus.textContent = `● Step ${Math.min(data.current_step_idx + 1, state.steps.length)}`;
    dom.elapsed.textContent = data.elapsed || '0 min 0 sec';
    const detections = data.detections || {};
    const entries = Object.entries(detections);
    dom.detectionList.innerHTML = entries.length
      ? entries.map(([label, confidence]) => `<span class="detection-chip">${label.replaceAll('_', ' ')} <b>${Math.round(confidence * 100)}%</b></span>`).join('')
      : '<span class="muted">No objects detected</span>';
  }

  function renderTimeline() {
    dom.stepList.innerHTML = state.steps.map((step, idx) => {
      let isDone = state.completedSteps.has(step.id);
      let isActive = idx === state.currentStepIdx;
      let status = state.stepStatuses[step.id];
      let cls = status === 'skipped' || status === 'out_of_sequence'
        ? 'warning'
        : (isDone ? 'completed' : (isActive ? 'active' : ''));
      let icon = status === 'skipped' ? '!' : (status === 'out_of_sequence' ? '!' : (isDone ? '✓' : (isActive ? '→' : '○')));

      return `
        <div class="step-row ${cls}">
          <div class="step-icon">${icon}</div>
          <div>
            <span class="step-code">${step.id}</span>
            <span class="step-name">${step.name}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  function renderLogs() {
    dom.logList.innerHTML = state.logs.slice().reverse().map(l => `
      <div class="log-item">
        <span class="log-time">${l.time}</span>
        <span class="log-event">
          <strong>${l.event}</strong>
          <small class="log-status ${l.status || 'info'}">${l.status || 'info'}</small>
          <em>${l.details || ''}</em>
        </span>
      </div>
    `).join('');
  }

  function addLog(event) {
    const time = new Date().toTimeString().split(' ')[0];
    state.logs.push({ time, event });
    renderLogs();
  }

  function speak(text) {
    dom.voiceText.textContent = `"${text}"`;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }
  }

  function setupListeners() {
    document.getElementById('btn-webcam')?.addEventListener('click', () => sendControl('webcam'));
    document.getElementById('btn-recording')?.addEventListener('click', () => sendControl('recording'));
    document.getElementById('btn-recording-2')?.addEventListener('click', () => sendControl('recording2'));
    document.getElementById('btn-quit')?.addEventListener('click', () => {
      if (confirm('Stop the camera/video pipeline?')) sendControl('quit');
    });
    dom.btnExportLog.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.logs, null, 2));
      const a = document.createElement('a');
      a.href = dataStr;
      a.download = `AstroHAR_Log_${Date.now()}.json`;
      a.click();
    });
  }

  function sendControl(action) {
    fetch('/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    }).catch(() => {});
  }

  function startCanvasOverlay() {
    const canvas = dom.cvCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function draw() {
      if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth || 640;
        canvas.height = canvas.clientHeight || 360;
      }
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Person Bounding Box
      ctx.strokeStyle = '#10B981'; ctx.lineWidth = 2;
      ctx.strokeRect(w*0.1, h*0.1, w*0.8, h*0.8);
      ctx.fillStyle = '#10B981'; ctx.font = '11px sans-serif';
      ctx.fillText('Person 0.98', w*0.1 + 4, h*0.1 + 14);

      // Red Box
      ctx.strokeStyle = '#EF4444';
      ctx.strokeRect(w*0.4, h*0.5, w*0.15, h*0.15);
      ctx.fillStyle = '#EF4444';
      ctx.fillText('Red Box 0.92', w*0.4 + 4, h*0.5 + 14);

      // Hand keypoint wrist
      const wx = w*0.45 + Math.sin(Date.now()/500)*12;
      const wy = h*0.53 + Math.cos(Date.now()/500)*8;
      ctx.fillStyle = '#00F2FE';
      ctx.beginPath(); ctx.arc(wx, wy, 5, 0, Math.PI*2); ctx.fill();
      ctx.fillText('Hand', wx - 12, wy + 18);

      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
