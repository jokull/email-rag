/**
 * Conversation Browser
 * Clean markdown-ish display of processed email conversations with similar conversation discovery
 */

import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { formatDistanceToNow } from 'date-fns'
import { 
  MessageSquare, 
  Users, 
  Clock, 
  TrendingUp, 
  Search,
  Filter,
  Star,
  ArrowRight,
  Sparkles,
  Bot,
  User,
  Building,
  Heart
} from 'lucide-react'

import { z } from '../lib/zero'
import { cn } from '../lib/utils'
import { searchConversations, ConversationWithSimilarity } from '../lib/conversation-api'

// Types for our enhanced conversation data
interface ConversationElement {
  id: string
  element_type: string
  content: string
  element_metadata: any
  sequence_number: number
}

interface ConversationTurn {
  id: string
  speaker_email: string
  speaker_name?: string
  turn_index: number
  turn_type: string
  clean_content: string
  word_count: number
  temporal_context: any
  created_at: string
}

interface ConversationDetail {
  id: string
  thread_id: string
  participants: string[]
  message_count: number
  conversation_type: string
  threading_confidence: number
  first_message_date: string
  last_message_date: string
  duration_days: number
  quality_metrics: any
  
  // Enhanced data from new pipeline
  classification?: {
    classification: string
    confidence: number
    sentiment_score: number
    importance_score: number
    commercial_score: number
    human_score: number
    personal_score: number
  }
  
  // Conversation turns with cleaned content
  turns: ConversationTurn[]
  
  // Similar conversations
  similar_conversations?: Array<{
    conversation_id: string
    similarity_score: number
    shared_participants: string[]
    topic_overlap: number
  }>
}

interface ConversationBrowserProps {
  className?: string
}

// Enhanced conversation fetching with vector similarity
const useConversations = (searchQuery: string, filters: any = {}) => {
  return useQuery({
    queryKey: ['conversations', searchQuery, filters],
    queryFn: async () => {
      return await searchConversations(searchQuery, filters)
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

// Helper function to find similar conversations
const findSimilarConversations = async (conversationId: string, participants: string[]) => {
  // In a real implementation, this would query the enhanced_embeddings table
  // and use vector similarity search. For now, we'll use participant overlap.
  
  const allConversations = await z.query.conversations.run()
  
  const similar = allConversations
    .filter(conv => conv.id !== conversationId)
    .map(conv => {
      const sharedParticipants = conv.participants.filter(p => participants.includes(p))
      const similarity = sharedParticipants.length / Math.max(conv.participants.length, participants.length)
      
      return {
        conversation_id: conv.id,
        similarity_score: similarity,
        shared_participants: sharedParticipants,
        topic_overlap: similarity // Simplified - would use actual topic modeling
      }
    })
    .filter(sim => sim.similarity_score > 0.3)
    .sort((a, b) => b.similarity_score - a.similarity_score)
    .slice(0, 5)
  
  return similar
}

// Conversation classification badge
const ClassificationBadge = ({ classification }: { classification: any }) => {
  if (!classification) return null
  
  const getClassificationStyle = (type: string) => {
    switch (type) {
      case 'human':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'promotional':
        return 'bg-orange-100 text-orange-800 border-orange-200'
      case 'transactional':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'automated':
        return 'bg-gray-100 text-gray-800 border-gray-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }
  
  const getClassificationIcon = (type: string) => {
    switch (type) {
      case 'human':
        return <User className="w-3 h-3" />
      case 'promotional':
        return <TrendingUp className="w-3 h-3" />
      case 'transactional':
        return <Building className="w-3 h-3" />
      case 'automated':
        return <Bot className="w-3 h-3" />
      default:
        return <MessageSquare className="w-3 h-3" />
    }
  }
  
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={cn(
        "inline-flex items-center gap-1 px-2 py-1 rounded-full border font-medium",
        getClassificationStyle(classification.classification)
      )}>
        {getClassificationIcon(classification.classification)}
        {classification.classification}
        <span className="opacity-75">
          {Math.round(classification.confidence * 100)}%
        </span>
      </span>
      
      {/* Sentiment indicator */}
      {classification.sentiment_score !== undefined && (
        <span className={cn(
          "inline-flex items-center gap-1 px-2 py-1 rounded-full border text-xs",
          classification.sentiment_score > 0.7 ? "bg-green-50 text-green-700 border-green-200" :
          classification.sentiment_score < 0.3 ? "bg-red-50 text-red-700 border-red-200" :
          "bg-yellow-50 text-yellow-700 border-yellow-200"
        )}>
          <Heart className="w-3 h-3" />
          {classification.sentiment_score > 0.7 ? 'Positive' :
           classification.sentiment_score < 0.3 ? 'Negative' : 'Neutral'}
        </span>
      )}
      
      {/* Importance indicator */}
      {classification.importance_score > 0.7 && (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full border text-xs bg-purple-50 text-purple-700 border-purple-200">
          <Star className="w-3 h-3" />
          Important
        </span>
      )}
    </div>
  )
}

// Conversation card component
const ConversationCard = ({ conversation }: { conversation: ConversationDetail }) => {
  const navigate = useNavigate()
  
  const duration = formatDistanceToNow(new Date(conversation.last_message_date), { addSuffix: true })
  const participantNames = conversation.participants.slice(0, 3).join(', ')
  const hasMoreParticipants = conversation.participants.length > 3
  
  // Get preview of the latest content
  const latestTurn = conversation.turns?.[conversation.turns.length - 1]
  const preview = latestTurn?.clean_content?.slice(0, 150) + (latestTurn?.clean_content?.length > 150 ? '...' : '')
  
  return (
    <div 
      className="group p-4 border border-gray-200 rounded-lg hover:border-gray-300 hover:shadow-sm transition-all cursor-pointer"
      onClick={() => navigate({ to: `/conversations/${conversation.id}` })}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <MessageSquare className="w-4 h-4 text-gray-500 flex-shrink-0" />
            <span className="text-sm font-medium text-gray-900 truncate">
              {conversation.conversation_type}
            </span>
            {conversation.threading_confidence > 0.9 && (
              <Sparkles className="w-3 h-3 text-yellow-500" title="High threading confidence" />
            )}
          </div>
          
          <ClassificationBadge classification={conversation.classification} />
        </div>
        
        <div className="text-xs text-gray-500 text-right">
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {duration}
          </div>
          <div className="mt-1">
            {conversation.message_count} messages
          </div>
        </div>
      </div>
      
      {/* Participants */}
      <div className="flex items-center gap-1 mb-3 text-sm text-gray-600">
        <Users className="w-3 h-3" />
        <span className="truncate">
          {participantNames}
          {hasMoreParticipants && (
            <span className="text-gray-400"> +{conversation.participants.length - 3} more</span>
          )}
        </span>
      </div>
      
      {/* Content preview */}
      {preview && (
        <div className="text-sm text-gray-700 mb-3 line-clamp-2">
          {preview}
        </div>
      )}
      
      {/* Similar conversations indicator */}
      {conversation.similar_conversations && conversation.similar_conversations.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <TrendingUp className="w-3 h-3" />
          <span>
            {conversation.similar_conversations.length} similar conversation{conversation.similar_conversations.length !== 1 ? 's' : ''}
          </span>
          <ArrowRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      )}
    </div>
  )
}

// Main conversation browser component
export const ConversationBrowser = ({ className }: ConversationBrowserProps) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState({
    classification: '',
    importance: '',
    participants: '',
    dateRange: ''
  })
  
  const { data: conversations, isLoading, error } = useConversations(searchQuery, filters)
  
  // Filter conversations based on search and filters
  const filteredConversations = useMemo(() => {
    if (!conversations) return []
    
    return conversations.filter(conv => {
      // Text search across participants and content
      if (searchQuery) {
        const searchLower = searchQuery.toLowerCase()
        const participantMatch = conv.participants.some(p => p.toLowerCase().includes(searchLower))
        const contentMatch = conv.turns?.some(turn => 
          turn.clean_content.toLowerCase().includes(searchLower)
        )
        if (!participantMatch && !contentMatch) return false
      }
      
      // Classification filter
      if (filters.classification && conv.classification?.classification !== filters.classification) {
        return false
      }
      
      // Importance filter
      if (filters.importance === 'high' && (!conv.classification || conv.classification.importance_score < 0.7)) {
        return false
      }
      
      return true
    })
  }, [conversations, searchQuery, filters])
  
  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-2">Failed to load conversations</div>
        <div className="text-sm text-gray-500">{error.message}</div>
      </div>
    )
  }
  
  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-900">Conversations</h1>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              {filteredConversations?.length || 0} conversations
            </span>
          </div>
        </div>
        
        {/* Search and filters */}
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search conversations, participants, or content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <select
            value={filters.classification}
            onChange={(e) => setFilters(prev => ({ ...prev, classification: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Types</option>
            <option value="human">Human</option>
            <option value="promotional">Promotional</option>
            <option value="transactional">Transactional</option>
            <option value="automated">Automated</option>
          </select>
          
          <select
            value={filters.importance}
            onChange={(e) => setFilters(prev => ({ ...prev, importance: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Importance</option>
            <option value="high">High Importance</option>
          </select>
        </div>
      </div>
      
      {/* Conversation list */}
      <div className="flex-1 overflow-auto">
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="text-gray-500">Loading conversations...</div>
          </div>
        ) : filteredConversations.length === 0 ? (
          <div className="p-8 text-center">
            <MessageSquare className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <div className="text-gray-500 mb-2">No conversations found</div>
            <div className="text-sm text-gray-400">
              {searchQuery || Object.values(filters).some(Boolean) 
                ? 'Try adjusting your search or filters'
                : 'Conversations will appear here as emails are processed'
              }
            </div>
          </div>
        ) : (
          <div className="p-6 space-y-4">
            {filteredConversations.map((conversation) => (
              <ConversationCard 
                key={conversation.id} 
                conversation={conversation}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}