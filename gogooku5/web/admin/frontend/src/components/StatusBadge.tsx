interface StatusBadgeProps {
  status: 'ok' | 'healthy' | 'error' | 'warning' | 'available' | 'active' | 'deprecated'
}

const statusColors: Record<string, string> = {
  ok: 'bg-green-100 text-green-800',
  healthy: 'bg-green-100 text-green-800',
  available: 'bg-blue-100 text-blue-800',
  active: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
  warning: 'bg-yellow-100 text-yellow-800',
  deprecated: 'bg-gray-100 text-gray-800',
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const colorClass = statusColors[status] || 'bg-gray-100 text-gray-800'

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${colorClass}`}>
      {status}
    </span>
  )
}
