import { createFileRoute } from '@tanstack/react-router'
import { SemanticSearch } from '../components/semantic-search'

export const Route = createFileRoute('/search')({
  component: Search,
})

function Search() {
  return (
    <div className="p-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Semantic Search</h1>
        <p className="text-muted-foreground">
          Search through your email conversations using natural language
        </p>
      </div>
      <SemanticSearch />
    </div>
  )
}