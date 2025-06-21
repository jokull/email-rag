import { useState } from 'react'
import { IconSearch, IconLoader } from 'justd-icons'
import { cn } from '../lib/utils'

interface SearchResult {
  threadId: number
  subject: string
  participants: string[]
  matchedChunk: string
  similarity: number
  lastMessageDate: string
}

export function SemanticSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setIsLoading(true)
    setHasSearched(true)

    try {
      // For now, this is a placeholder - in the real implementation,
      // this would call your semantic search API endpoint
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      })

      if (response.ok) {
        const searchResults = await response.json()
        setResults(searchResults)
      } else {
        // Placeholder results for demo
        setResults([
          {
            threadId: 1,
            subject: 'Project Discussion',
            participants: ['john@example.com', 'jane@example.com'],
            matchedChunk: 'This matches your search query about project updates...',
            similarity: 0.89,
            lastMessageDate: '2024-01-15T10:30:00Z',
          },
          {
            threadId: 2,
            subject: 'Meeting Follow-up',
            participants: ['alice@example.com'],
            matchedChunk: 'Another relevant chunk that relates to your search...',
            similarity: 0.76,
            lastMessageDate: '2024-01-14T14:20:00Z',
          },
        ])
      }
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-4xl">
      {/* Search Form */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="relative">
          <IconSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your email conversations using natural language..."
            className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className={cn(
            "mt-3 px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium",
            "hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            "flex items-center gap-2"
          )}
        >
          {isLoading ? (
            <>
              <IconLoader className="h-4 w-4 animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <IconSearch className="h-4 w-4" />
              Search
            </>
          )}
        </button>
      </form>

      {/* Search Results */}
      {hasSearched && (
        <div>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <IconLoader className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">
                Search Results ({results.length})
              </h2>
              {results.map((result, index) => (
                <div key={index} className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-medium">
                      {result.subject || 'No Subject'}
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="bg-primary/10 text-primary px-2 py-1 rounded text-xs">
                        {Math.round(result.similarity * 100)}% match
                      </span>
                      <span>
                        {new Date(result.lastMessageDate).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  
                  <div className="text-sm text-muted-foreground mb-3">
                    Participants: {result.participants.join(', ')}
                  </div>
                  
                  <div className="bg-muted/30 p-3 rounded text-sm">
                    <div className="font-medium mb-1">Relevant excerpt:</div>
                    <p className="text-muted-foreground">{result.matchedChunk}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <IconSearch className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No results found for "{query}"</p>
              <p className="text-sm">Try rephrasing your search or using different keywords.</p>
            </div>
          )}
        </div>
      )}

      {/* Search Tips */}
      {!hasSearched && (
        <div className="bg-muted/30 rounded-lg p-6">
          <h3 className="font-semibold mb-3">Search Tips</h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Use natural language: "emails about project deadlines"</li>
            <li>• Search by topic: "vacation requests" or "meeting schedules"</li>
            <li>• Find conversations: "discussions with John about budget"</li>
            <li>• Semantic search understands context and meaning</li>
          </ul>
        </div>
      )}
    </div>
  )
}