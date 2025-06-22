import { useState, useMemo, useRef, useEffect, useCallback } from 'react'
import { useSearchParams } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import { zero } from '../lib/zero'
import { cn } from '../lib/utils'
import { ConversationRow } from './conversation-row'
import { ConversationFilters } from './conversation-filters'
import { formatDistanceToNow } from 'date-fns'
import { 
  IconSearch, 
  IconLoader, 
  IconMail, 
  IconFilter,
  IconX,
  IconChevronDown 
} from 'justd-icons'

interface ConversationListPageProps {
  onConversationSelect?: (conversationId: string) => void
  selectedConversationId?: string
}

export function ConversationListPage({ 
  onConversationSelect, 
  selectedConversationId 
}: ConversationListPageProps) {
  const [searchParams, setSearchParams] = useSearchParams()
  const [isSearchMode, setIsSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [limit, setLimit] = useState(50)
  
  const listRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Parse URL parameters for filters and sorting
  const filters = useMemo(() => ({
    classification: searchParams.get('classification') || 'human',
    participantFilter: searchParams.get('participants') || '',
    domainFilter: searchParams.get('domain') || '',
    minHumanScore: Number(searchParams.get('humanScore')) || 0.7,
    minPersonalScore: Number(searchParams.get('personalScore')) || 0,
    minRelevanceScore: Number(searchParams.get('relevanceScore')) || 0,
    conversationType: searchParams.get('type') || '',
    dateRange: searchParams.get('dateRange') || '30d',
    sortBy: (searchParams.get('sortBy') as 'lastMessage' | 'firstMessage' | 'messageCount') || 'lastMessage',
    sortOrder: (searchParams.get('sortOrder') as 'asc' | 'desc') || 'desc',
    search: searchParams.get('search') || ''
  }), [searchParams])

  // Zero query for conversations with dynamic filtering
  const conversationsQuery = zero.query.conversations
    .related('classification', q => q)
    .related('turns', q => q.limit(3).orderBy('turnIndex', 'asc'))

  // Apply filters dynamically
  let filteredQuery = conversationsQuery

  // Classification filter
  if (filters.classification !== 'all') {
    filteredQuery = filteredQuery.related('classification', q => 
      q.where('classification', filters.classification)
        .where('humanScore', '>=', filters.minHumanScore)
        .where('personalScore', '>=', filters.minPersonalScore)
        .where('relevanceScore', '>=', filters.minRelevanceScore)
    )
  }

  // Conversation type filter
  if (filters.conversationType) {
    filteredQuery = filteredQuery.where('conversationType', filters.conversationType)
  }

  // Date range filter
  const dateThreshold = useMemo(() => {
    const now = new Date()
    switch (filters.dateRange) {
      case '7d': return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString()
      case '30d': return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString()
      case '90d': return new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000).toISOString()
      default: return '2020-01-01T00:00:00Z'
    }
  }, [filters.dateRange])

  filteredQuery = filteredQuery.where('lastMessageDate', '>=', dateThreshold)

  // Apply sorting
  filteredQuery = filteredQuery.orderBy(
    filters.sortBy === 'lastMessage' ? 'lastMessageDate' :
    filters.sortBy === 'firstMessage' ? 'firstMessageDate' : 'messageCount',
    filters.sortOrder
  ).limit(limit)

  const conversations = filteredQuery.useQuery()

  // Filter by search query client-side for now
  const filteredConversations = useMemo(() => {
    if (!filters.search && !filters.participantFilter && !filters.domainFilter) {
      return conversations
    }

    return conversations.filter(conv => {
      const participants = Array.isArray(conv.participants) ? conv.participants : []
      
      // Search in participants or content
      if (filters.search) {
        const searchLower = filters.search.toLowerCase()
        const matchesParticipants = participants.some(p => 
          p.toLowerCase().includes(searchLower)
        )
        if (!matchesParticipants) return false
      }

      // Participant filter
      if (filters.participantFilter) {
        const participantLower = filters.participantFilter.toLowerCase()
        const hasParticipant = participants.some(p =>
          p.toLowerCase().includes(participantLower)
        )
        if (!hasParticipant) return false
      }

      // Domain filter
      if (filters.domainFilter) {
        const domainLower = filters.domainFilter.toLowerCase()
        const hasDomain = participants.some(p => {
          const domain = p.split('@')[1]?.toLowerCase()
          return domain?.includes(domainLower)
        })
        if (!hasDomain) return false
      }

      return true
    })
  }, [conversations, filters.search, filters.participantFilter, filters.domainFilter])

  // Virtualization setup
  const virtualizer = useVirtualizer({
    count: filteredConversations.length,
    getScrollElement: () => listRef.current,
    estimateSize: () => 120, // Estimated row height
    overscan: 5,
  })

  // Update search params
  const updateFilter = useCallback((key: string, value: string | number) => {
    const newParams = new URLSearchParams(searchParams)
    if (value) {
      newParams.set(key, value.toString())
    } else {
      newParams.delete(key)
    }
    setSearchParams(newParams)
  }, [searchParams, setSearchParams])

  // Handle search
  const handleSearch = useCallback((query: string) => {
    updateFilter('search', query)
  }, [updateFilter])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && !isSearchMode) {
        e.preventDefault()
        setIsSearchMode(true)
        searchInputRef.current?.focus()
      } else if (e.key === 'Escape' && isSearchMode) {
        setIsSearchMode(false)
        setSearchQuery('')
        handleSearch('')
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isSearchMode, handleSearch])

  // Handle scroll to load more
  const handleScroll = useCallback(() => {
    if (!listRef.current) return
    
    const { scrollTop, scrollHeight, clientHeight } = listRef.current
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - 100
    
    if (isNearBottom && filteredConversations.length >= limit - 10) {
      setLimit(prev => prev + 50)
    }
  }, [filteredConversations.length, limit])

  useEffect(() => {
    const listElement = listRef.current
    if (listElement) {
      listElement.addEventListener('scroll', handleScroll)
      return () => listElement.removeEventListener('scroll', handleScroll)
    }
  }, [handleScroll])

  // Statistics
  const stats = useMemo(() => {
    const total = filteredConversations.length
    const withClassifications = filteredConversations.filter(c => c.classification).length
    const avgConfidence = filteredConversations.reduce((sum, c) => 
      sum + (c.threadingConfidence || 0), 0) / (total || 1)
    
    return { total, withClassifications, avgConfidence }
  }, [filteredConversations])

  return (
    <div className="flex flex-col h-full">
      {/* Header with search and filters */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold">Conversations</h1>
              <p className="text-sm text-muted-foreground">
                {stats.total} conversations • {stats.withClassifications} classified • 
                {(stats.avgConfidence * 100).toFixed(0)}% avg confidence
              </p>
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={cn(
                  "px-3 py-2 rounded-lg border text-sm font-medium transition-colors",
                  showFilters 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-background hover:bg-muted"
                )}
              >
                <IconFilter className="h-4 w-4 mr-2" />
                Filters
                <IconChevronDown className={cn(
                  "h-4 w-4 ml-2 transition-transform",
                  showFilters && "rotate-180"
                )} />
              </button>
            </div>
          </div>

          {/* Search */}
          <div className="relative">
            {isSearchMode ? (
              <div className="relative">
                <IconSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleSearch(searchQuery)
                      setIsSearchMode(false)
                    }
                  }}
                  placeholder="Search conversations, participants..."
                  className="w-full pl-10 pr-10 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                  autoFocus
                />
                <button
                  onClick={() => {
                    setIsSearchMode(false)
                    setSearchQuery('')
                    handleSearch('')
                  }}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <IconX className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsSearchMode(true)}
                className="w-full flex items-center px-3 py-2 border rounded-lg text-muted-foreground hover:bg-muted transition-colors"
              >
                <IconSearch className="h-4 w-4 mr-2" />
                Search conversations... (Press / to search)
              </button>
            )}
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="px-4 pb-4 border-t">
            <ConversationFilters
              filters={filters}
              onFilterChange={updateFilter}
            />
          </div>
        )}
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-hidden">
        {filteredConversations.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <IconMail className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium mb-2">No conversations found</p>
              <p className="text-sm">
                {filters.search ? 
                  `No results for "${filters.search}"` :
                  "Try adjusting your filters or check back later"
                }
              </p>
            </div>
          </div>
        ) : (
          <div
            ref={listRef}
            className="h-full overflow-auto"
            style={{ contain: 'strict' }}
          >
            <div
              style={{
                height: `${virtualizer.getTotalSize()}px`,
                width: '100%',
                position: 'relative',
              }}
            >
              {virtualizer.getVirtualItems().map((virtualItem) => {
                const conversation = filteredConversations[virtualItem.index]
                if (!conversation) return null

                return (
                  <div
                    key={virtualItem.key}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: `${virtualItem.size}px`,
                      transform: `translateY(${virtualItem.start}px)`,
                    }}
                  >
                    <ConversationRow
                      conversation={conversation}
                      isSelected={selectedConversationId === conversation.id}
                      onClick={() => onConversationSelect?.(conversation.id)}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Loading indicator */}
      {filteredConversations.length >= limit - 10 && (
        <div className="flex justify-center p-4 border-t">
          <IconLoader className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      )}
    </div>
  )
}