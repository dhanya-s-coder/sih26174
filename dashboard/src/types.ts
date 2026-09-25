export type StepStatus = 'completed' | 'current' | 'pending' | 'skipped' | 'out_of_sequence' | 'failed'

export interface ExperimentStep {
  id: string
  position: number
  name: string
  instruction: string
  status: StepStatus
  outcome: string
}

export interface ActivityItem {
  time: string
  event: string
  category: string
  status: string
}

export interface RuntimeState {
  system: string
  experiment: { name: string; version: string }
  progress: {
    current_index: number
    total: number
    current_step: ExperimentStep | null
    completed: ExperimentStep[]
    steps: ExperimentStep[]
    complete: boolean
    duration_seconds: number | null
    error_count: number
  }
  action: string
  assistant: { message: string; status: string }
  alert: {
    message: string
    level: string
    timestamp: number
    expected: string
    detected: string
  } | null
  health: {
    camera: string
    detection: string
    hand_tracking: string
    experiment_tracking: string
    voice_assistant: string
  }
  camera: { stream_url: string; frame_id: number | null }
  activity: ActivityItem[]
  updated_at: string
}

export interface WebSocketEnvelope {
  type: 'runtime_update'
  data: RuntimeState
}
