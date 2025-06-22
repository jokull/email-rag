import { useMemo } from 'react'
import { cn } from '../lib/utils'
import { 
  IconFilter,
  IconX,
  IconUser,
  IconCalendar,
  IconStar,
  IconBrain,
  IconHeart,
  IconTarget,
  IconMessage,
  IconSettings
} from 'justd-icons'

interface ConversationFiltersProps {
  filters: {
    classification: string
    participantFilter: string
    domainFilter: string
    minHumanScore: number
    minPersonalScore: number
    minRelevanceScore: number
    conversationType: string
    dateRange: string
    sortBy: string
    sortOrder: string
    search: string
  }
  onFilterChange: (key: string, value: string | number) => void
}

export function ConversationFilters({ filters, onFilterChange }: ConversationFiltersProps) {
  // Classification options
  const classificationOptions = [
    { value: 'all', label: 'All Classifications', color: 'bg-gray-100 text-gray-700' },
    { value: 'human', label: 'Human Conversations', color: 'bg-blue-100 text-blue-700' },
    { value: 'promotional', label: 'Promotional', color: 'bg-orange-100 text-orange-700' },
    { value: 'transactional', label: 'Transactional', color: 'bg-green-100 text-green-700' },
    { value: 'automated', label: 'Automated', color: 'bg-red-100 text-red-700' }
  ]

  // Conversation type options
  const conversationTypeOptions = [
    { value: '', label: 'All Types' },
    { value: 'question_answer', label: 'Q&A' },
    { value: 'discussion', label: 'Discussion' },
    { value: 'announcement', label: 'Announcement' },
    { value: 'negotiation', label: 'Negotiation' },
    { value: 'status_update', label: 'Status Update' },
    { value: 'social', label: 'Social' }
  ]

  // Date range options
  const dateRangeOptions = [
    { value: '7d', label: 'Last 7 days' },
    { value: '30d', label: 'Last 30 days' },
    { value: '90d', label: 'Last 90 days' },
    { value: 'all', label: 'All time' }
  ]

  // Sort options
  const sortOptions = [
    { value: 'lastMessage', label: 'Last Message' },
    { value: 'firstMessage', label: 'First Message' },
    { value: 'messageCount', label: 'Message Count' }
  ]

  // Count active filters
  const activeFilterCount = useMemo(() => {
    let count = 0
    if (filters.classification !== 'human') count++
    if (filters.participantFilter) count++
    if (filters.domainFilter) count++
    if (filters.minHumanScore !== 0.7) count++
    if (filters.minPersonalScore !== 0) count++
    if (filters.minRelevanceScore !== 0) count++
    if (filters.conversationType) count++
    if (filters.dateRange !== '30d') count++
    if (filters.search) count++
    return count
  }, [filters])

  // Reset all filters
  const resetFilters = () => {
    onFilterChange('classification', 'human')
    onFilterChange('participantFilter', '')
    onFilterChange('domainFilter', '')
    onFilterChange('humanScore', 0.7)
    onFilterChange('personalScore', 0)
    onFilterChange('relevanceScore', 0)
    onFilterChange('type', '')
    onFilterChange('dateRange', '30d')
    onFilterChange('search', '')
  }

  return (
    <div className="space-y-6 py-4">
      {/* Filter Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <IconFilter className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium text-sm">Filters</span>
          {activeFilterCount > 0 && (
            <span className="px-2 py-0.5 bg-primary/10 text-primary text-xs rounded-full">
              {activeFilterCount} active
            </span>
          )}
        </div>
        {activeFilterCount > 0 && (
          <button
            onClick={resetFilters}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
          >
            <IconX className="h-3 w-3" />
            Reset all
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Classification Filter */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <IconBrain className="h-4 w-4 text-blue-500" />
            Classification
          </label>
          <div className="grid grid-cols-1 gap-2">
            {classificationOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => onFilterChange('classification', option.value)}
                className={cn(
                  "px-3 py-2 rounded-lg text-xs font-medium transition-colors text-left",
                  filters.classification === option.value
                    ? option.color
                    : "bg-muted/50 text-muted-foreground hover:bg-muted"
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Conversation Type Filter */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <IconMessage className="h-4 w-4 text-green-500" />
            Conversation Type
          </label>
          <select
            value={filters.conversationType}
            onChange={(e) => onFilterChange('type', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {conversationTypeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Date Range Filter */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <IconCalendar className="h-4 w-4 text-purple-500" />
            Date Range
          </label>
          <select
            value={filters.dateRange}
            onChange={(e) => onFilterChange('dateRange', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {dateRangeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Score Filters */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium flex items-center gap-2">
          <IconStar className="h-4 w-4 text-yellow-500" />
          AI Classification Scores
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Human Score */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground flex items-center justify-between">
              <span className="flex items-center gap-1">
                <IconBrain className="h-3 w-3 text-blue-500" />
                Human Score
              </span>
              <span>{(filters.minHumanScore * 100).toFixed(0)}%+</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minHumanScore}
              onChange={(e) => onFilterChange('humanScore', Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Personal Score */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground flex items-center justify-between">
              <span className="flex items-center gap-1">
                <IconHeart className="h-3 w-3 text-red-500" />
                Personal Score
              </span>
              <span>{(filters.minPersonalScore * 100).toFixed(0)}%+</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minPersonalScore}
              onChange={(e) => onFilterChange('personalScore', Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Relevance Score */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground flex items-center justify-between">
              <span className="flex items-center gap-1">
                <IconTarget className="h-3 w-3 text-purple-500" />
                Relevance Score
              </span>
              <span>{(filters.minRelevanceScore * 100).toFixed(0)}%+</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minRelevanceScore}
              onChange={(e) => onFilterChange('relevanceScore', Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Participant Filters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <IconUser className="h-4 w-4 text-green-500" />
            Participant Filter
          </label>
          <input
            type="text"
            value={filters.participantFilter}
            onChange={(e) => onFilterChange('participants', e.target.value)}
            placeholder="Filter by participant email..."
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <IconSettings className="h-4 w-4 text-orange-500" />
            Domain Filter
          </label>
          <input
            type="text"
            value={filters.domainFilter}
            onChange={(e) => onFilterChange('domain', e.target.value)}
            placeholder="Filter by domain (e.g., company.com)..."
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>
      </div>

      {/* Sort Options */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Sort By</label>
          <select
            value={filters.sortBy}
            onChange={(e) => onFilterChange('sortBy', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {sortOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Sort Order</label>
          <select
            value={filters.sortOrder}
            onChange={(e) => onFilterChange('sortOrder', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            <option value="desc">Newest First</option>
            <option value="asc">Oldest First</option>
          </select>
        </div>
      </div>
    </div>
  )
}