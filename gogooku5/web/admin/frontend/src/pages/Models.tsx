import { Link } from 'react-router-dom'
import { useModels } from '../hooks/useModels'
import { Card } from '../components/Card'
import { StatusBadge } from '../components/StatusBadge'

export function Models() {
  const { data: models, isLoading, error } = useModels()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading models...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Error loading models: {String(error)}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Models</h1>
        <span className="text-gray-500">{models?.length || 0} models</span>
      </div>

      <Card>
        {models && models.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b bg-gray-50">
                  <th className="text-left py-3 px-4">Name</th>
                  <th className="text-left py-3 px-4">Size</th>
                  <th className="text-left py-3 px-4">Created</th>
                  <th className="text-left py-3 px-4">Status</th>
                  <th className="text-left py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <tr
                    key={model.name}
                    className={`border-b hover:bg-gray-50 ${
                      index === 0 ? 'bg-blue-50' : ''
                    }`}
                  >
                    <td className="py-3 px-4">
                      <Link
                        to={`/models/${model.name}`}
                        className="text-blue-600 hover:underline font-medium"
                      >
                        {model.name}
                      </Link>
                      {index === 0 && (
                        <span className="ml-2 text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded">
                          Latest
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4">{model.size_mb.toFixed(1)} MB</td>
                    <td className="py-3 px-4">
                      {new Date(model.created_at * 1000).toLocaleString('ja-JP')}
                    </td>
                    <td className="py-3 px-4">
                      <StatusBadge status={model.status as 'available'} />
                    </td>
                    <td className="py-3 px-4">
                      <Link
                        to={`/models/${model.name}`}
                        className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
                      >
                        View Metrics
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No models found</p>
        )}
      </Card>
    </div>
  )
}
