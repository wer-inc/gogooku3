interface MetricCardProps {
  label: string
  value: string | number
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
}

export function MetricCard({ label, value, subValue, trend }: MetricCardProps) {
  const trendColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-500',
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold mt-1">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </p>
      {subValue && (
        <p className={`text-sm mt-1 ${trend ? trendColors[trend] : 'text-gray-500'}`}>
          {subValue}
        </p>
      )}
    </div>
  )
}
