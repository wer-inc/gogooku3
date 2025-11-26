// API Types

export interface User {
  id: number
  email: string
  username: string
  is_active: boolean
  is_admin: boolean
  created_at: string
  updated_at: string | null
}

export interface UserCreate {
  email: string
  username: string
  password: string
  is_active?: boolean
  is_admin?: boolean
}

export interface MLModel {
  name: string
  path: string
  size_mb: number
  created_at: number
  status: string
  description?: string
}

export interface MLModelMetrics {
  model_name: string
  rank_ic: number
  p_at_k: number
  p_at_k_rand: number
  delta_p_at_k: number
  ndcg: number
  ndcg_rand: number
  delta_ndcg: number
  topk_overlap: number
  spread: number
  wil: number
  validation_days: number
}

export interface StockPrediction {
  code: string
  name?: string
  rank: number
  score: number
  sector?: string
}

export interface Prediction {
  id: number
  prediction_date: string
  model_name: string
  horizon_days: number
  top_k: number
  stocks: StockPrediction[]
  created_at: string
  rank_ic?: number
  p_at_k?: number
}

export interface HealthStatus {
  status: string
  components?: {
    api: string
    database: string
    ml_models: string
  }
}
