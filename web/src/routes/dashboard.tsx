import { createFileRoute } from '@tanstack/react-router'
import { StatusDashboard } from '../components/status-dashboard'
import { Navigation } from '../components/navigation'

export const Route = createFileRoute('/dashboard')({
  component: Dashboard,
})

function Dashboard() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <StatusDashboard />
    </div>
  )
}