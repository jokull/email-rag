import { createFileRoute } from '@tanstack/react-router'
import { ProcessingDashboard } from '../components/processing-dashboard'

export const Route = createFileRoute('/processing')({
  component: Processing,
})

function Processing() {
  return (
    <div className="p-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Processing Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor AI processing progress, budgets, and queue status
        </p>
      </div>
      <ProcessingDashboard />
    </div>
  )
}