import { Card } from '../components/Card'

export function Predictions() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Predictions</h1>

      <Card>
        <div className="text-center py-12">
          <p className="text-gray-500 text-lg mb-4">
            Prediction feature coming soon
          </p>
          <p className="text-gray-400">
            This section will display model predictions and allow running new predictions.
          </p>
        </div>
      </Card>
    </div>
  )
}
