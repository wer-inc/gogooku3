import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { MLModel, MLModelMetrics } from '../api/types'

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => api.get('models').json<MLModel[]>(),
  })
}

export function useModel(modelName: string) {
  return useQuery({
    queryKey: ['models', modelName],
    queryFn: () => api.get(`models/${modelName}`).json<MLModel>(),
    enabled: !!modelName,
  })
}

export function useModelMetrics(modelName: string) {
  return useQuery({
    queryKey: ['models', modelName, 'metrics'],
    queryFn: () => api.get(`models/${modelName}/metrics`).json<MLModelMetrics>(),
    enabled: !!modelName,
  })
}

export function useActivateModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (modelName: string) =>
      api.post(`models/${modelName}/activate`).json(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}
