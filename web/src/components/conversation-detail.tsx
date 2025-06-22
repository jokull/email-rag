/**
 * Conversation Detail View
 * Displays cleaned email content in markdown-ish format with similar conversations
 */

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams, Link } from '@tanstack/react-router'
import { formatDistanceToNow, format } from 'date-fns'
import { 
  ArrowLeft,
  Users, 
  Clock, 
  MessageSquare,
  Star,
  TrendingUp,
  ExternalLink,
  Copy,
  Share,
  MoreHorizontal,
  User,
  Bot,
  Building,
  Heart,
  Sparkles,
  Link as LinkIcon
} from 'lucide-react'

import { z } from '../lib/zero'
import { cn } from '../lib/utils'
import { getConversationDetail, ConversationWithSimilarity } from '../lib/conversation-api'

// Enhanced types for conversation detail - matches database schema
interface EmailElement {
  id: string
  email_id: string
  element_id: string
  element_type: string
  content: string
  markdown_content?: string  // Clean markdown version of content
  element_metadata: any
  coordinates?: any
  page_number: number
  sequence_number: number
  parent_element_id?: string
  extraction_confidence: number
  processing_method: string
  is_cleaned: boolean
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
  content_analysis: any
  created_at: string
}

interface ConversationData {
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
  temporal_patterns: any
  conversation_flow: any
  
  // Enhanced data
  classification?: {
    classification: string
    confidence: number
    sentiment_score: number
    importance_score: number
    commercial_score: number
    human_score: number
    personal_score: number
    relevance_score: number
  }
  
  turns: ConversationTurn[]
  elements: EmailElement[]
  similar_conversations: Array<{
    conversation_id: string
    similarity_score: number
    shared_participants: string[]
    topic_overlap: number
    conversation_type: string
    last_message_date: string
    message_count: number
  }>
}

// Use conversation detail data
const useConversationDetail = (conversationId: string) => {
  return useQuery({
    queryKey: ['conversation-detail', conversationId],
    queryFn: async () => {
      const conversation = await getConversationDetail(conversationId)
      if (!conversation) throw new Error('Conversation not found')
      return conversation
    },
    enabled: !!conversationId,
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

// Enhanced similar conversation finder
const findSimilarConversationsWithDetails = async (conversationId: string, participants: string[]) => {
  // Get all conversations for similarity comparison
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
        topic_overlap: similarity, // Would use actual topic modeling in production
        conversation_type: conv.conversation_type,
        last_message_date: conv.last_message_date,
        message_count: conv.message_count
      }
    })
    .filter(sim => sim.similarity_score > 0.2)
    .sort((a, b) => b.similarity_score - a.similarity_score)
    .slice(0, 8)
  
  return similar
}

// Content formatter for markdown-ish display
const formatEmailContent = (content: string, elements?: EmailElement[]) => {
  if (!content) return content
  
  // If we have processed elements with markdown, use those instead
  if (elements && elements.length > 0) {
    // Check if any elements have markdown content
    const markdownElements = elements.filter(el => el.markdown_content)
    if (markdownElements.length > 0) {
      // Combine markdown elements in sequence order
      const combinedMarkdown = markdownElements
        .sort((a, b) => a.sequence_number - b.sequence_number)
        .map(el => el.markdown_content)
        .join('')
      
      if (combinedMarkdown.trim()) {
        return convertMarkdownToHTML(combinedMarkdown)
      }
    }
  }
  
  // Fallback to basic markdown-ish formatting
  let formatted = content
  
  // Format URLs as clickable links
  formatted = formatted.replace(
    /https?:\/\/[^\s]+/g, 
    (url) => `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">${url}</a>`
  )
  
  // Format email addresses
  formatted = formatted.replace(
    /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    (email) => `<span class="font-mono text-sm bg-gray-100 px-1 rounded">${email}</span>`
  )
  
  // Format phone numbers (basic pattern)
  formatted = formatted.replace(
    /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
    (phone) => `<span class="font-mono text-sm bg-gray-100 px-1 rounded">${phone}</span>`
  )
  
  // Format quoted text (lines starting with >)
  formatted = formatted.replace(
    /^&gt;(.*)$/gm,
    '<div class="border-l-4 border-gray-300 pl-4 text-gray-600 italic bg-gray-50 py-1">$1</div>'
  )
  
  return formatted
}

// Simple markdown to HTML converter for clean display
const convertMarkdownToHTML = (markdown: string): string => {
  let html = markdown
  
  // Headers
  html = html.replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold mb-2 text-gray-900">$1</h1>')
  html = html.replace(/^## (.+)$/gm, '<h2 class="text-lg font-semibold mb-2 text-gray-800">$1</h2>')
  html = html.replace(/^### (.+)$/gm, '<h3 class="text-base font-medium mb-1 text-gray-700">$1</h3>')
  
  // Lists
  html = html.replace(/^- (.+)$/gm, '<li class="ml-4 mb-1">• $1</li>')
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li class="ml-4 mb-1">$1. $2</li>')
  
  // Code blocks
  html = html.replace(/```\n([\s\S]*?)\n```/g, '<pre class="bg-gray-100 p-3 rounded-lg text-sm font-mono overflow-x-auto">$1</pre>')
  
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 rounded text-sm font-mono">$1</code>')
  
  // Links (already in markdown format [text](url))
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">$1</a>')
  
  // Paragraphs (double newlines)
  html = html.replace(/\n\n/g, '</p><p class="mb-3">')
  html = `<p class="mb-3">${html}</p>`
  
  // Single newlines to <br>
  html = html.replace(/\n/g, '<br>')
  
  // Clean up empty paragraphs
  html = html.replace(/<p class="mb-3"><\/p>/g, '')
  
  return html
}

// Speaker avatar component
const SpeakerAvatar = ({ email, name }: { email: string; name?: string }) => {
  const initials = name ? name.split(' ').map(n => n[0]).join('').toUpperCase() 
                        : email.split('@')[0].slice(0, 2).toUpperCase()
  
  const colors = [
    'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-pink-500', 
    'bg-indigo-500', 'bg-yellow-500', 'bg-red-500', 'bg-teal-500'
  ]
  
  const colorIndex = email.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length
  
  return (
    <div className={cn(
      "w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-medium",
      colors[colorIndex]
    )}>
      {initials}
    </div>
  )
}

// Conversation turn component
const ConversationTurnCard = ({ turn, elements, isLast }: { turn: ConversationTurn; elements?: EmailElement[]; isLast: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const timeAgo = formatDistanceToNow(new Date(turn.created_at), { addSuffix: true })
  
  // Get elements for this specific email/turn
  const turnElements = elements?.filter(el => el.email_id === turn.email_id) || []
  const formattedContent = formatEmailContent(turn.clean_content, turnElements)
  
  return (
    <div className={cn(
      "relative border-l-2 border-gray-200 pl-6 pb-8",
      isLast && "border-l-transparent"
    )}>
      {/* Timeline dot */}
      <div className="absolute -left-2 top-0 w-4 h-4 bg-white border-2 border-blue-500 rounded-full"></div>
      
      {/* Speaker header */}
      <div className="flex items-center gap-3 mb-3">
        <SpeakerAvatar email={turn.speaker_email} name={turn.speaker_name} />
        <div className="flex-1">
          <div className="font-medium text-gray-900">
            {turn.speaker_name || turn.speaker_email.split('@')[0]}
          </div>
          <div className="text-sm text-gray-500">
            {turn.speaker_email} • {timeAgo}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="bg-gray-100 px-2 py-1 rounded-full">
            {turn.turn_type}
          </span>
          <span>
            {turn.word_count} words
          </span>
        </div>
      </div>
      
      {/* Content */}
      <div className={cn(
        "bg-white border border-gray-200 rounded-lg p-4",
        !isExpanded && "cursor-pointer hover:border-gray-300"
      )}
      onClick={() => !isExpanded && setIsExpanded(true)}>
        {isExpanded ? (
          <div 
            className="prose prose-sm max-w-none text-gray-700 leading-relaxed"
            dangerouslySetInnerHTML={{ __html: formattedContent }}
          />
        ) : (
          <div className="text-gray-600">
            {formattedContent.slice(0, 150)}...
            <span className="text-blue-600 ml-2">Click to expand</span>
          </div>
        )}
        
        {isExpanded && formattedContent.length > 300 && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              setIsExpanded(false)
            }}
            className="mt-3 text-sm text-gray-500 hover:text-gray-700"
          >
            Collapse
          </button>
        )}
      </div>
    </div>
  )
}

// Similar conversation card
const SimilarConversationCard = ({ conversation }: { conversation: any }) => {
  const timeAgo = formatDistanceToNow(new Date(conversation.last_message_date), { addSuffix: true })
  
  return (
    <Link
      to={`/conversations/${conversation.conversation_id}`}
      className="block p-3 border border-gray-200 rounded-lg hover:border-gray-300 hover:shadow-sm transition-all"
    >
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1">
          <div className="text-sm font-medium text-gray-900 truncate">
            {conversation.conversation_type}
          </div>
          <div className="text-xs text-gray-500">
            {conversation.message_count} messages • {timeAgo}
          </div>
        </div>
        <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
          {Math.round(conversation.similarity_score * 100)}% match
        </div>
      </div>
      
      {conversation.shared_participants.length > 0 && (
        <div className="text-xs text-gray-600">
          Shared: {conversation.shared_participants.slice(0, 2).join(', ')}
          {conversation.shared_participants.length > 2 && ` +${conversation.shared_participants.length - 2}`}
        </div>
      )}
    </Link>
  )
}

// Classification summary component
const ClassificationSummary = ({ classification }: { classification: any }) => {
  if (!classification) return null
  
  const scores = [
    { label: 'Importance', value: classification.importance_score, color: 'text-purple-600' },
    { label: 'Human', value: classification.human_score, color: 'text-blue-600' },
    { label: 'Personal', value: classification.personal_score, color: 'text-green-600' },
    { label: 'Sentiment', value: classification.sentiment_score, color: 'text-pink-600' },
  ]
  
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">AI Classification</h3>
      
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{classification.classification}</span>
          <span className="text-xs text-gray-500">
            {Math.round(classification.confidence * 100)}% confidence
          </span>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs">
          {scores.map(score => (
            <div key={score.label} className="flex justify-between">
              <span className="text-gray-600">{score.label}</span>
              <span className={score.color}>
                {Math.round(score.value * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Main conversation detail component
export const ConversationDetail = () => {
  const { conversationId } = useParams({ from: '/conversations/$conversationId' })
  const { data: conversation, isLoading, error } = useConversationDetail(conversationId)
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading conversation...</div>
      </div>
    )
  }
  
  if (error || !conversation) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-red-600 mb-2">Failed to load conversation</div>
          <div className="text-sm text-gray-500">
            {error?.message || 'Conversation not found'}
          </div>
        </div>
      </div>
    )
  }
  
  const duration = format(new Date(conversation.last_message_date), 'MMM d, yyyy')
  const durationDays = conversation.duration_days
  
  return (
    <div className="flex h-full">
      {/* Main conversation view */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center gap-4 mb-4">
            <Link to="/conversations" className="text-gray-500 hover:text-gray-700">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-gray-900">
                {conversation.conversation_type}
              </h1>
              <div className="flex items-center gap-4 text-sm text-gray-500 mt-1">
                <span className="flex items-center gap-1">
                  <Users className="w-4 h-4" />
                  {conversation.participants.length} participants
                </span>
                <span className="flex items-center gap-1">
                  <MessageSquare className="w-4 h-4" />
                  {conversation.message_count} messages
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {durationDays > 0 ? `${durationDays} days` : 'Same day'}
                </span>
                {conversation.threading_confidence > 0.9 && (
                  <span className="flex items-center gap-1 text-yellow-600">
                    <Sparkles className="w-4 h-4" />
                    High confidence threading
                  </span>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <button className="p-2 text-gray-500 hover:text-gray-700 border border-gray-300 rounded-lg">
                <Share className="w-4 h-4" />
              </button>
              <button className="p-2 text-gray-500 hover:text-gray-700 border border-gray-300 rounded-lg">
                <MoreHorizontal className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          {/* Participants */}
          <div className="flex items-center gap-2">
            {conversation.participants.slice(0, 6).map((email, idx) => (
              <SpeakerAvatar key={email} email={email} />
            ))}
            {conversation.participants.length > 6 && (
              <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-xs text-gray-600">
                +{conversation.participants.length - 6}
              </div>
            )}
            <div className="ml-3 text-sm text-gray-600">
              {conversation.participants.slice(0, 3).join(', ')}
              {conversation.participants.length > 3 && ` and ${conversation.participants.length - 3} others`}
            </div>
          </div>
        </div>
        
        {/* Conversation turns */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-4xl mx-auto">
            {conversation.turns.map((turn, idx) => (
              <ConversationTurnCard 
                key={turn.id} 
                turn={turn} 
                isLast={idx === conversation.turns.length - 1}
              />
            ))}
          </div>
        </div>
      </div>
      
      {/* Sidebar */}
      <div className="w-80 border-l border-gray-200 bg-gray-50 overflow-auto">
        <div className="p-6 space-y-6">
          {/* Classification */}
          <ClassificationSummary classification={conversation.classification} />
          
          {/* Similar conversations */}
          {conversation.similar_conversations.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Similar Conversations
              </h3>
              <div className="space-y-2">
                {conversation.similar_conversations.map((similar) => (
                  <SimilarConversationCard 
                    key={similar.conversation_id} 
                    conversation={similar}
                  />
                ))}
              </div>
            </div>
          )}
          
          {/* Conversation metadata */}
          <div>
            <h3 className="text-sm font-medium text-gray-900 mb-3">Details</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Started</span>
                <span className="text-gray-900">
                  {format(new Date(conversation.first_message_date), 'MMM d, yyyy')}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Last activity</span>
                <span className="text-gray-900">{duration}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Threading confidence</span>
                <span className="text-gray-900">
                  {Math.round(conversation.threading_confidence * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}