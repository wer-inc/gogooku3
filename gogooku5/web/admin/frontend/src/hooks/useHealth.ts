import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { HealthStatus } from '../api/types'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => api.get('health').json<HealthStatus>(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}

export function useDetailedHealth() {
  return useQuery({
    queryKey: ['health', 'detailed'],
    queryFn: () => api.get('health/detailed').json<HealthStatus>(),
    refetchInterval: 30000,
  })
}
