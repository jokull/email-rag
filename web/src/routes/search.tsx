import { createFileRoute } from '@tanstack/react-router'
import { SemanticSearch } from '../components/semantic-search'
import { Navigation } from '../components/navigation'

export const Route = createFileRoute('/search')({
  component: Search,
})

function Search() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="max-w-7xl mx-auto p-6">
        <SemanticSearch />
      </div>
    </div>
  )
}