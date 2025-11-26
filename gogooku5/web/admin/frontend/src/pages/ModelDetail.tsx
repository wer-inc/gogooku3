import { useParams, Link } from 'react-router-dom'
import { useModel, useModelMetrics, useActivateModel } from '../hooks/useModels'
import { Card } from '../components/Card'
import { MetricCard } from '../components/MetricCard'
import { StatusBadge } from '../components/StatusBadge'

export function ModelDetail() {
  const { modelName } = useParams<{ modelName: string }>()
  const { data: model, isLoading: modelLoading } = useModel(modelName || '')
  const { data: metrics, isLoading: metricsLoading } = useModelMetrics(modelName || '')
  const activateMutation = useActivateModel()

  if (modelLoading || metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading model details...</p>
      </div>
    )
  }

  if (!model) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Model not found</p>
        <Link to="/models" className="text-blue-600 hover:underline mt-2 inline-block">
          ← Back to Models
        </Link>
      </div>
    )
  }

  const handleActivate = () => {
    if (modelName) {
      activateMutation.mutate(modelName)
    }
  }

  return (
    <div>
      <div className="flex items-center gap-4 mb-6">
        <Link to="/models" className="text-gray-500 hover:text-gray-700">
          ← Models
        </Link>
        <h1 className="text-2xl font-bold">{model.name}</h1>
        <StatusBadge status={model.status as 'available'} />
      </div>

      {/* Model Info */}
      <Card title="Model Information" className="mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500">Size</p>
            <p className="font-medium">{model.size_mb.toFixed(1)} MB</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Created</p>
            <p className="font-medium">
              {new Date(model.created_at * 1000).toLocaleString('ja-JP')}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Status</p>
            <p className="font-medium">{model.status}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Validation Days</p>
            <p className="font-medium">{metrics?.validation_days || 'N/A'}</p>
          </div>
        </div>
        <div className="mt-4">
          <button
            onClick={handleActivate}
            disabled={activateMutation.isPending}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:bg-gray-400"
          >
            {activateMutation.isPending ? 'Activating...' : 'Activate Model'}
          </button>
        </div>
      </Card>

      {/* Metrics */}
      {metrics && (
        <>
          <h2 className="text-xl font-semibold mb-4">Performance Metrics (20d Horizon)</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard
              label="RankIC"
              value={metrics.rank_ic}
              trend={metrics.rank_ic > 0.05 ? 'up' : metrics.rank_ic > 0 ? 'neutral' : 'down'}
            />
            <MetricCard
              label="P@K"
              value={metrics.p_at_k}
              subValue={`Δ${metrics.delta_p_at_k > 0 ? '+' : ''}${(metrics.delta_p_at_k * 100).toFixed(1)}%`}
              trend={metrics.delta_p_at_k > 0.05 ? 'up' : 'neutral'}
            />
            <MetricCard
              label="NDCG"
              value={metrics.ndcg}
              subValue={`Δ${metrics.delta_ndcg > 0 ? '+' : ''}${(metrics.delta_ndcg * 100).toFixed(1)}%`}
              trend={metrics.delta_ndcg > 0 ? 'up' : 'neutral'}
            />
            <MetricCard
              label="TopK Overlap"
              value={metrics.topk_overlap}
            />
          </div>

          <Card title="Detailed Metrics">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">RankIC</p>
                <p className="text-xl font-bold text-blue-600">
                  {metrics.rank_ic > 0 ? '+' : ''}{metrics.rank_ic.toFixed(4)}
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">P@K (Model)</p>
                <p className="text-xl font-bold">{metrics.p_at_k.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">P@K (Random)</p>
                <p className="text-xl font-bold text-gray-500">{metrics.p_at_k_rand.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">NDCG (Model)</p>
                <p className="text-xl font-bold">{metrics.ndcg.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">NDCG (Random)</p>
                <p className="text-xl font-bold text-gray-500">{metrics.ndcg_rand.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">Spread</p>
                <p className="text-xl font-bold">{metrics.spread.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-500">WIL</p>
                <p className="text-xl font-bold">{metrics.wil.toFixed(4)}</p>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
