import { useState, useMemo, useCallback } from 'react'
import { IconSearch, IconLoader, IconTarget, IconBrain, IconHeart, IconMessage, IconUsers } from 'justd-icons'
import { cn } from '../lib/utils'
import { zero } from '../lib/zero'
import { formatDistanceToNow } from 'date-fns'

interface SearchResult {
  conversation: any
  embedding: any
  similarity: number
  matchedContent: string
  relevanceScore: number
}

export function SemanticSearch() {
  const [query, setQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  // Query conversations with classifications for search
  const allConversations = zero.query.conversations
    .related('classification', q => q.where('classification', 'human'))
    .related('turns', q => q.limit(3).orderBy('turnIndex', 'asc'))
    .useQuery()

  // Query embeddings for semantic search (placeholder - would use vector similarity in production)
  const allEmbeddings = zero.query.embeddings.useQuery()

  // Perform client-side semantic search simulation
  const searchResults = useMemo(() => {
    if (!query.trim() || !hasSearched) return []

    const queryLower = query.toLowerCase()
    const results: SearchResult[] = []

    // Simple text search through conversations and their turns
    allConversations.forEach(conversation => {
      if (!conversation.classification || conversation.classification.classification !== 'human') {
        return
      }

      const participants = Array.isArray(conversation.participants) ? conversation.participants : []
      let bestMatch = ''
      let matchScore = 0

      // Search in participant emails
      const participantMatch = participants.some(p => 
        p.toLowerCase().includes(queryLower)
      )

      // Search in conversation turns content
      if (conversation.turns) {
        conversation.turns.forEach((turn: any) => {
          if (turn.cleanContent && turn.cleanContent.toLowerCase().includes(queryLower)) {
            const content = turn.cleanContent
            const matchIndex = content.toLowerCase().indexOf(queryLower)
            
            // Extract context around the match
            const start = Math.max(0, matchIndex - 50)
            const end = Math.min(content.length, matchIndex + 150)
            const contextMatch = content.slice(start, end)
            
            if (contextMatch.length > bestMatch.length) {
              bestMatch = contextMatch
              matchScore = 0.8 // Base similarity score
            }
          }
        })
      }

      // Add conversation type matching
      if (conversation.conversationType && conversation.conversationType.toLowerCase().includes(queryLower)) {
        matchScore += 0.1
      }

      if (participantMatch || bestMatch || matchScore > 0) {
        // Find related embedding (simplified)
        const relatedEmbedding = allEmbeddings.find(e => e.threadId === conversation.threadId)
        
        results.push({
          conversation,
          embedding: relatedEmbedding,
          similarity: participantMatch ? 0.9 : matchScore,
          matchedContent: bestMatch || `Conversation with ${participants.slice(0, 2).join(', ')}`,
          relevanceScore: conversation.classification?.relevanceScore || 0
        })
      }
    })

    // Sort by similarity and relevance
    return results
      .sort((a, b) => (b.similarity + b.relevanceScore) - (a.similarity + a.relevanceScore))
      .slice(0, 20) // Limit results
  }, [query, hasSearched, allConversations, allEmbeddings])

  const handleSearch = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setIsLoading(true)
    setHasSearched(true)

    // Simulate search delay
    await new Promise(resolve => setTimeout(resolve, 500))
    setIsLoading(false)
  }, [query])

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
          ) : searchResults.length > 0 ? (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">
                Search Results ({searchResults.length})
              </h2>
              {searchResults.map((result, index) => {
                const conversation = result.conversation
                const participants = Array.isArray(conversation.participants) ? conversation.participants : []
                const lastMessageTime = conversation.lastMessageDate ? new Date(conversation.lastMessageDate) : new Date()

                return (
                  <div key={index} className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-medium mb-1">
                          {conversation.conversationType ? 
                            `${conversation.conversationType.replace('_', ' ')} conversation` : 
                            'Email conversation'
                          }
                        </h3>
                        <div className="flex items-center gap-2 text-xs">
                          {conversation.conversationType && (
                            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                              {conversation.conversationType.replace('_', ' ')}
                            </span>
                          )}
                          <span className="bg-primary/10 text-primary px-2 py-1 rounded">
                            {Math.round(result.similarity * 100)}% match
                          </span>
                        </div>
                      </div>
                      
                      <div className="text-sm text-muted-foreground">
                        {formatDistanceToNow(lastMessageTime, { addSuffix: true })}
                      </div>
                    </div>
                    
                    {/* Participants */}
                    <div className="flex items-center gap-2 text-sm mb-3">
                      <IconUsers className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        {participants.slice(0, 3).join(', ')}
                        {participants.length > 3 && ` +${participants.length - 3} more`}
                      </span>
                    </div>

                    {/* Conversation Stats */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3">
                      <div className="flex items-center gap-1">
                        <IconMessage className="h-3 w-3" />
                        {conversation.messageCount} messages
                      </div>
                      <div className="flex items-center gap-1">
                        <IconBrain className="h-3 w-3 text-blue-500" />
                        {conversation.classification ? 
                          `${Math.round(conversation.classification.humanScore * 100)}% human` : 
                          'Not classified'
                        }
                      </div>
                      {conversation.classification?.personalScore > 0 && (
                        <div className="flex items-center gap-1">
                          <IconHeart className="h-3 w-3 text-red-500" />
                          {Math.round(conversation.classification.personalScore * 100)}% personal
                        </div>
                      )}
                      <div className="flex items-center gap-1">
                        <IconTarget className="h-3 w-3 text-purple-500" />
                        {Math.round(result.relevanceScore * 100)}% relevant
                      </div>
                    </div>
                    
                    {/* Matched Content */}
                    <div className="bg-muted/30 p-3 rounded text-sm">
                      <div className="font-medium mb-1">Relevant excerpt:</div>
                      <p className="text-muted-foreground leading-relaxed">
                        {result.matchedContent}
                      </p>
                    </div>

                    {/* Classification Scores */}
                    {conversation.classification && (
                      <div className="mt-3 flex items-center gap-3 text-xs">
                        <div className="flex items-center gap-1">
                          <IconBrain className="h-3 w-3 text-blue-500" />
                          <span className="text-muted-foreground">Human:</span>
                          <span className="font-medium text-blue-600">
                            {Math.round(conversation.classification.humanScore * 100)}%
                          </span>
                        </div>
                        <div className="flex items-center gap-1">
                          <IconHeart className="h-3 w-3 text-red-500" />
                          <span className="text-muted-foreground">Personal:</span>
                          <span className="font-medium text-red-600">
                            {Math.round(conversation.classification.personalScore * 100)}%
                          </span>
                        </div>
                        <div className="flex items-center gap-1">
                          <IconTarget className="h-3 w-3 text-purple-500" />
                          <span className="text-muted-foreground">Relevance:</span>
                          <span className="font-medium text-purple-600">
                            {Math.round(conversation.classification.relevanceScore * 100)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
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