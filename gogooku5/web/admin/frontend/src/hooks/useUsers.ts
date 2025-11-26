import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { User, UserCreate } from '../api/types'

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => api.get('users').json<User[]>(),
  })
}

export function useUser(userId: number) {
  return useQuery({
    queryKey: ['users', userId],
    queryFn: () => api.get(`users/${userId}`).json<User>(),
    enabled: !!userId,
  })
}

export function useCreateUser() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (user: UserCreate) =>
      api.post('users', { json: user }).json<User>(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
    },
  })
}

export function useDeleteUser() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (userId: number) => api.delete(`users/${userId}`).json(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
    },
  })
}
