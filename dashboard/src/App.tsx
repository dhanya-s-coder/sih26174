import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Check, CheckCircle2, Circle, Clock3, Radio, RefreshCw, RotateCcw, Satellite, Wifi, X } from 'lucide-react'
import type { ActivityItem, RuntimeState, WebSocketEnvelope } from './types'

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN ?? ''
const EMPTY: RuntimeState = {
  system: 'connecting',
  experiment: { name: 'Connecting to HAR runtime…', version: '' },
  progress: { current_index: 0, total: 0, current_step: null, completed: [], steps: [], complete: false, duration_seconds: null, error_count: 0 },
  action: '',
  assistant: { message: '', status: 'N/A' },
  alert: null,
  health: { camera: 'N/A', detection: 'N/A', hand_tracking: 'N/A', experiment_tracking: 'N/A', voice_assistant: 'N/A' },
  camera: { stream_url: '', frame_id: null },
  activity: [],
  updated_at: '',
}

function socketUrl() {
  if (import.meta.env.VITE_API_ORIGIN) {
    const api = new URL(import.meta.env.VITE_API_ORIGIN)
    api.protocol = api.protocol === 'https:' ? 'wss:' : 'ws:'
    api.pathname = '/ws'
    return api.toString()
  }
  return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`
}

function useRuntime() {
  const [state, setState] = useState<RuntimeState>(EMPTY)
  const [connected, setConnected] = useState(false)

  const poll = useCallback(async () => {
    try {
      const response = await fetch(`${API_ORIGIN}/api/state`, { cache: 'no-store' })
      if (response.ok) setState(await response.json() as RuntimeState)
    } catch {
      setConnected(false)
    }
  }, [])

  useEffect(() => {
    let socket: WebSocket | undefined
    let retry: number | undefined
    let alive = true
    const connect = () => {
      if (!alive) return
      socket = new WebSocket(socketUrl())
      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketEnvelope
          if (message.type === 'runtime_update') setState(message.data)
        } catch {
          // Ignore malformed messages; REST remains the fallback source.
        }
      }
      socket.onclose = () => {
        setConnected(false)
        retry = window.setTimeout(connect, 1500)
      }
      socket.onerror = () => socket?.close()
    }
    void poll()
    connect()
    const fallback = window.setInterval(() => void poll(), 2500)
    return () => {
      alive = false
      socket?.close()
      window.clearInterval(fallback)
      if (retry) window.clearTimeout(retry)
    }
  }, [poll])

  return { state, setState, connected }
}

function healthLabel(value: string) {
  const normalized = value.toUpperCase()
  if (['ONLINE', 'RUNNING', 'READY', 'ACTIVE', 'PLAYING'].includes(normalized)) return 'good'
  if (['ERROR', 'FAILED'].includes(normalized)) return 'bad'
  return 'waiting'
}

function activityTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
}

function App() {
  const { state, setState, connected } = useRuntime()
  const [showLogs, setShowLogs] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [resetError, setResetError] = useState('')
  const steps = state.progress.steps
  const current = state.progress.current_step
  const wrongStep = Boolean(state.alert && ['warning', 'critical', 'error'].includes(state.alert.level.toLowerCase()) && Date.now() / 1000 - state.alert.timestamp < 15)
  const cameraError = state.health.camera.toUpperCase() === 'ERROR'
  const subsystemError = [state.health.detection, state.health.hand_tracking, state.health.experiment_tracking, state.health.voice_assistant]
    .some((status) => status.toUpperCase() === 'ERROR')
  const systemText = state.system === 'error' || cameraError ? 'SYSTEM ERROR'
    : !connected ? 'API OFFLINE'
      : subsystemError ? 'SYSTEM DEGRADED'
      : state.progress.complete ? 'EXPERIMENT COMPLETE'
        : state.health.camera === 'ONLINE' ? 'SYSTEM ONLINE'
          : state.health.camera === 'STOPPED' ? 'SOURCE ENDED' : 'CONNECTING'
  const healthItems = useMemo(() => [
    ['Camera', state.health.camera],
    ['Detection', state.health.detection],
    ['Hand tracking', state.health.hand_tracking],
    ['Experiment tracking', state.health.experiment_tracking],
    ['Voice assistant', state.health.voice_assistant],
  ] as const, [state.health])

  const resetExperiment = async () => {
    if (!window.confirm('Reset the current experiment?')) return
    setResetting(true)
    setResetError('')
    try {
      const response = await fetch(`${API_ORIGIN}/api/experiment/reset`, { method: 'POST' })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail ?? 'The experiment could not be reset.')
      }
      setState(await response.json() as RuntimeState)
    } catch (error) {
      setResetError(error instanceof Error ? error.message : 'The experiment could not be reset.')
    } finally {
      setResetting(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><Satellite size={23} strokeWidth={1.7} /></div>
        <div className="brand-copy">
          <div className="brand-title">AI SPACE EXPERIMENT ASSISTANT</div>
          <div className="brand-subtitle">Autonomous Experiment Validation <span className="brand-separator">/</span> ISRO · SIH 26174</div>
        </div>
        <div className={`system-pill ${systemText === 'SYSTEM ONLINE' || systemText === 'EXPERIMENT COMPLETE' ? 'is-good' : systemText === 'SYSTEM ERROR' || systemText === 'API OFFLINE' ? 'is-bad' : 'is-waiting'}`}>
          <span className="status-dot" />{systemText}
        </div>
      </header>

      <section className="main-grid">
        <div className="left-column">
          <section className="card camera-card">
            <div className="section-heading">
              <div><span className="eyebrow">MISSION VIEW</span><h2>Live camera</h2></div>
              <div className="live-tag"><span className="live-dot" />LIVE</div>
            </div>
            <div className="video-stage">
              {state.camera.stream_url ? (
                <img className="camera-feed" src={state.camera.stream_url} alt="Processed HAR camera feed" />
              ) : <div className="camera-placeholder"><Radio size={28} /><span>Waiting for processed camera feed</span></div>}
              <div className="camera-overlay-label"><span className="viewfinder-icon" /> HAR PROCESSED FEED</div>
            </div>
            <div className="camera-caption"><span><Wifi size={14} /> Local onboard processing</span><span>Frame {state.camera.frame_id ?? '—'}</span></div>
          </section>

          <section className="card activity-card">
            <div className="section-heading activity-heading">
              <div><span className="eyebrow">MISSION RECORD</span><h2>Experiment activity</h2></div>
              <button className="text-button" onClick={() => setShowLogs(true)}>View full logs <span>↗</span></button>
            </div>
            <ActivityList items={state.activity.slice(0, 5)} />
          </section>
        </div>

        <div className="right-column">
          <section className="card progress-card">
            <div className="section-heading">
              <div><span className="eyebrow">SEQUENCE</span><h2>Experiment progress</h2></div>
              <div className="progress-actions"><div className="step-total">{String(state.progress.total).padStart(2, '0')} <span>STEPS</span></div>
                <button className="reset-button" onClick={resetExperiment} disabled={resetting || !connected} title="Reset the active experiment">
                  <RotateCcw size={13} />{resetting ? 'RESETTING…' : 'RESET EXPERIMENT'}
                </button>
              </div>
            </div>
            <div className="experiment-name">{state.experiment.name}</div>
            <div className="progress-list">
              {steps.length ? steps.map((step) => (
                <div className={`progress-item state-${step.status}`} key={step.id}>
                  <div className="progress-icon">
                    {step.status === 'completed' ? <Check size={17} /> : step.status === 'current' ? <span className="current-arrow">→</span> : step.status === 'pending' ? <Circle size={15} /> : <AlertTriangle size={16} />}
                  </div>
                  <div className="progress-text"><span className="step-number">{step.position}.</span> {step.name}</div>
                  {step.status === 'current' && <span className="current-tag">CURRENT</span>}
                  {step.status === 'completed' && <span className="done-tag">DONE</span>}
                </div>
              )) : <div className="empty-state">Protocol steps will appear when the runtime connects.</div>}
            </div>
            {resetError && <div className="reset-error" role="alert">{resetError}</div>}
          </section>

          <section className={`card current-card ${state.progress.complete ? 'complete-card' : ''}`}>
            <div className="section-heading compact-heading"><div><span className="eyebrow">NOW</span><h2>{state.progress.complete ? 'Experiment complete' : 'Current step'}</h2></div>
              {state.progress.complete ? <CheckCircle2 className="complete-icon" size={23} /> : <span className="step-fraction">{current ? `STEP ${current.position} / ${state.progress.total}` : 'READY'}</span>}
            </div>
            <div className="current-step-name">{state.progress.complete ? 'All steps completed' : current?.name ?? 'Waiting for experiment'}</div>
            <div className="in-progress-label"><span className="status-dot" />{state.progress.complete ? 'SUCCESS' : current ? 'IN PROGRESS' : 'AWAITING START'}</div>
          </section>

          <section className="card action-card">
            <div className="section-heading compact-heading"><div><span className="eyebrow">INTERACTION</span><h2>Current action</h2></div><span className="hand-glyph">✋</span></div>
            <div className="action-text">{state.action ? humanize(state.action) : 'Waiting for action…'}</div>
          </section>

          <section className="card assistant-card">
            <div className="section-heading compact-heading"><div><span className="eyebrow">VOICE GUIDANCE</span><h2>Assistant</h2></div><div className="sound-wave"><i /><i /><i /><i /><i /></div></div>
            <div className="assistant-message">{state.assistant.message ? `“${state.assistant.message}”` : 'Waiting for instruction…'}</div>
          </section>

          {wrongStep && state.alert && <section className="warning-card">
            <div className="warning-title"><AlertTriangle size={21} /> WRONG STEP</div>
            <div className="warning-values"><div><span>EXPECTED</span><strong>{state.alert.expected || 'Unavailable'}</strong></div><div><span>DETECTED</span><strong>{humanize(state.alert.detected) || 'Unavailable'}</strong></div></div>
            <p>{state.alert.message}</p>
          </section>}
        </div>
      </section>

      {state.progress.complete && <section className="completion-banner"><div className="completion-icon"><CheckCircle2 size={25} /></div><div><h2>Experiment complete</h2><p>✓ All {state.progress.total} experiment steps completed</p></div><div className="completion-meta"><span><Clock3 size={15} />{state.progress.duration_seconds == null ? 'Duration unavailable' : `${Math.round(state.progress.duration_seconds)} sec`}</span><span>{state.progress.error_count} recorded {state.progress.error_count === 1 ? 'error' : 'errors'}</span></div></section>}

      <footer className="status-footer">
        <div className="footer-title">SYSTEM STATUS</div>
        <div className="health-list">{healthItems.map(([name, status]) => <div className={`health-item ${healthLabel(status)}`} key={name}><span className="status-dot" />{name}</div>)}</div>
        <div className="footer-brand">LOCAL · OFFLINE <span>•</span> SIH 26174</div>
      </footer>

      {showLogs && <LogsModal items={state.activity} onClose={() => setShowLogs(false)} />}
    </main>
  )
}

function humanize(value: string) {
  return value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (letter: string) => letter.toUpperCase())
}

function ActivityList({ items }: { items: ActivityItem[] }) {
  if (!items.length) return <div className="empty-state activity-empty">No experiment events received yet.</div>
  return <div className="activity-list">{items.map((item, index) => <div className="activity-row" key={`${item.time}-${index}`}>
    <span className="activity-time">{activityTime(item.time)}</span><span className={`activity-status ${item.status}`}>
      {item.status === 'warning' || item.status === 'error' || item.status === 'critical' ? <AlertTriangle size={14} /> : <Check size={14} />}
    </span><span className="activity-event">{item.event}</span><span className="activity-category">{item.category}</span>
  </div>)}</div>
}

function LogsModal({ items, onClose }: { items: ActivityItem[]; onClose: () => void }) {
  const [liveItems, setLiveItems] = useState(items)
  const [stepRecords, setStepRecords] = useState<ActivityItem[]>([])

  useEffect(() => {
    let alive = true
    fetch(`${API_ORIGIN}/api/logs?limit=500`, { cache: 'no-store' })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Log request failed')))
      .then((payload: { items?: ActivityItem[]; step_records?: ActivityItem[] }) => {
        if (alive) {
          setLiveItems(payload.items ?? [])
          setStepRecords(payload.step_records ?? [])
        }
      })
      .catch(() => undefined)
    return () => { alive = false }
  }, [])

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="logs-modal" role="dialog" aria-modal="true" aria-labelledby="logs-title">
      <header className="modal-header"><div><span className="eyebrow">LOCAL EVENT HISTORY</span><h2 id="logs-title">Experiment activity</h2></div><button className="icon-button" onClick={onClose} aria-label="Close logs"><X size={19} /></button></header>
      <div className="modal-body">
        <div className="modal-section-title">Recent runtime events</div>
        <ActivityList items={liveItems} />
        <div className="modal-section-title step-record-heading">Recorded experiment steps</div>
        <ActivityList items={stepRecords} />
      </div>
      <div className="modal-footer"><RefreshCw size={14} /> Live events from the active HAR runtime</div>
    </section>
  </div>
}

export default App
