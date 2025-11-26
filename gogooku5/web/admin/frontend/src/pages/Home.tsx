import { useDetailedHealth } from '../hooks/useHealth'
import { useModels } from '../hooks/useModels'
import { Card } from '../components/Card'
import { StatusBadge } from '../components/StatusBadge'
import { MetricCard } from '../components/MetricCard'
import { Link } from 'react-router-dom'

export function Home() {
  const { data: health, isLoading: healthLoading } = useDetailedHealth()
  const { data: models, isLoading: modelsLoading } = useModels()

  const latestModel = models?.[0]

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      {/* System Status */}
      <Card title="System Status" className="mb-6">
        {healthLoading ? (
          <p className="text-gray-500">Loading...</p>
        ) : health ? (
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <span className="text-gray-600">API:</span>
              <StatusBadge status={health.components?.api as 'healthy' || 'ok'} />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-600">Database:</span>
              <StatusBadge status={health.components?.database as 'healthy' || 'ok'} />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-600">ML Models:</span>
              <StatusBadge status={health.components?.ml_models as 'healthy' || 'ok'} />
            </div>
          </div>
        ) : (
          <p className="text-red-500">Unable to fetch health status</p>
        )}
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="Total Models"
          value={models?.length || 0}
        />
        <MetricCard
          label="Active Model"
          value={latestModel?.name || 'N/A'}
        />
        <MetricCard
          label="Latest Model Size"
          value={latestModel ? `${latestModel.size_mb.toFixed(1)} MB` : 'N/A'}
        />
        <MetricCard
          label="Models Status"
          value={modelsLoading ? 'Loading...' : 'Available'}
        />
      </div>

      {/* Recent Models */}
      <Card title="Recent Models">
        {modelsLoading ? (
          <p className="text-gray-500">Loading models...</p>
        ) : models && models.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Name</th>
                  <th className="text-left py-2">Size</th>
                  <th className="text-left py-2">Created</th>
                  <th className="text-left py-2">Status</th>
                  <th className="text-left py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {models.slice(0, 5).map((model) => (
                  <tr key={model.name} className="border-b hover:bg-gray-50">
                    <td className="py-2">
                      <Link
                        to={`/models/${model.name}`}
                        className="text-blue-600 hover:underline"
                      >
                        {model.name}
                      </Link>
                    </td>
                    <td className="py-2">{model.size_mb.toFixed(1)} MB</td>
                    <td className="py-2">
                      {new Date(model.created_at * 1000).toLocaleString('ja-JP')}
                    </td>
                    <td className="py-2">
                      <StatusBadge status={model.status as 'available'} />
                    </td>
                    <td className="py-2">
                      <Link
                        to={`/models/${model.name}`}
                        className="text-sm text-blue-600 hover:underline"
                      >
                        View Details
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500">No models found</p>
        )}
      </Card>
    </div>
  )
}
