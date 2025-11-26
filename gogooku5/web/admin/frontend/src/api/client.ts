import ky from 'ky'

export const api = ky.create({
  prefixUrl: (import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/api',
  timeout: 30000,
  hooks: {
    beforeError: [
      async (error) => {
        const { response } = error
        if (response) {
          try {
            const body = await response.json()
            error.message = body.detail || error.message
          } catch {
            // Ignore JSON parse errors
          }
        }
        return error
      },
    ],
  },
})

// Type-safe API response types
export interface ApiResponse<T> {
  data: T
  status: number
}
