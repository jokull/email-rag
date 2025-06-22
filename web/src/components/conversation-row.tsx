import { useMemo } from 'react'
import { formatDistanceToNow, format } from 'date-fns'
import { cn } from '../lib/utils'
import { 
  IconUser, 
  IconClock, 
  IconMessage, 
  IconStar,
  IconTrendingUp,
  IconUsers,
  IconBrain,
  IconHeart,
  IconTarget
} from 'justd-icons'

interface ConversationRowProps {
  conversation: any // Type from Zero schema
  isSelected?: boolean
  onClick?: () => void
}

export function ConversationRow({ conversation, isSelected, onClick }: ConversationRowProps) {
  // Extract conversation metadata
  const {
    id,
    participants = [],
    messageCount,
    conversationType,
    threadingConfidence,
    firstMessageDate,
    lastMessageDate,
    durationDays,
    qualityMetrics = {},
    classification,
    turns = []
  } = conversation

  // Get first turn for preview content
  const firstTurn = turns[0]
  const lastTurn = turns[turns.length - 1]

  // Parse participants
  const participantList = Array.isArray(participants) ? participants : []
  const displayParticipants = participantList.slice(0, 3)
  const extraParticipants = Math.max(0, participantList.length - 3)

  // Get classification scores
  const classificationData = classification || {}
  const {
    humanScore = 0,
    personalScore = 0,
    relevanceScore = 0,
    shouldProcess = false
  } = classificationData

  // Format dates
  const lastMessageTime = lastMessageDate ? new Date(lastMessageDate) : new Date()
  const firstMessageTime = firstMessageDate ? new Date(firstMessageDate) : new Date()
  
  const timeAgo = formatDistanceToNow(lastMessageTime, { addSuffix: true })
  const isRecent = (Date.now() - lastMessageTime.getTime()) < 24 * 60 * 60 * 1000

  // Get conversation type display
  const typeDisplay = useMemo(() => {
    switch (conversationType) {
      case 'question_answer': return { label: 'Q&A', color: 'text-blue-600', bg: 'bg-blue-50' }
      case 'discussion': return { label: 'Discussion', color: 'text-green-600', bg: 'bg-green-50' }
      case 'announcement': return { label: 'Announcement', color: 'text-purple-600', bg: 'bg-purple-50' }
      case 'negotiation': return { label: 'Negotiation', color: 'text-orange-600', bg: 'bg-orange-50' }
      case 'status_update': return { label: 'Status', color: 'text-gray-600', bg: 'bg-gray-50' }
      case 'social': return { label: 'Social', color: 'text-pink-600', bg: 'bg-pink-50' }
      default: return { label: 'Conversation', color: 'text-gray-600', bg: 'bg-gray-50' }
    }
  }, [conversationType])

  // Get content preview
  const contentPreview = useMemo(() => {
    if (firstTurn?.cleanContent) {
      return firstTurn.cleanContent.slice(0, 120) + (firstTurn.cleanContent.length > 120 ? '...' : '')
    }
    return 'No content available'
  }, [firstTurn])

  // Determine conversation subject/title
  const conversationTitle = useMemo(() => {
    if (turns.length > 0) {
      const emails = turns.map(t => t.emailId).filter(Boolean)
      // Could query related emails for subject, but for now use conversation type
      return `${typeDisplay.label} with ${participantList.length} participant${participantList.length !== 1 ? 's' : ''}`
    }
    return 'Email conversation'
  }, [turns, typeDisplay.label, participantList.length])

  // Speaker attribution for preview
  const speakerInfo = useMemo(() => {
    if (firstTurn?.speakerEmail) {
      const name = firstTurn.speakerName || firstTurn.speakerEmail.split('@')[0]
      const domain = firstTurn.speakerEmail.split('@')[1]
      return { name, email: firstTurn.speakerEmail, domain }
    }
    return null
  }, [firstTurn])

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-4 border-b cursor-pointer transition-all duration-200 hover:bg-muted/50",
        isSelected && "bg-muted border-l-4 border-l-primary",
        isRecent && "bg-accent/5"
      )}
    >
      <div className="space-y-3">
        {/* Header Row */}
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-medium text-sm truncate">
                {conversationTitle}
              </h3>
              <div className={cn(
                "px-2 py-0.5 rounded-full text-xs font-medium",
                typeDisplay.color,
                typeDisplay.bg
              )}>
                {typeDisplay.label}
              </div>
              {shouldProcess && (
                <div className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
                  RAG
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="whitespace-nowrap">{timeAgo}</span>
            {isRecent && (
              <div className="w-2 h-2 bg-primary rounded-full" />
            )}
          </div>
        </div>

        {/* Participants Row */}
        <div className="flex items-center gap-2 text-sm">
          <IconUsers className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex items-center gap-1 min-w-0 flex-1">
            {displayParticipants.map((participant, index) => {
              const [name, domain] = participant.split('@')
              return (
                <span
                  key={participant}
                  className="text-muted-foreground truncate"
                  title={participant}
                >
                  {name}
                  {index < displayParticipants.length - 1 && ', '}
                </span>
              )
            })}
            {extraParticipants > 0 && (
              <span className="text-muted-foreground">
                +{extraParticipants}
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <IconMessage className="h-3 w-3" />
              {messageCount}
            </div>
            <div className="flex items-center gap-1">
              <IconClock className="h-3 w-3" />
              {durationDays > 0 ? `${durationDays}d` : '<1d'}
            </div>
          </div>
        </div>

        {/* Content Preview */}
        {speakerInfo && (
          <div className="text-sm">
            <div className="flex items-start gap-2">
              <div className="flex-shrink-0 w-6 h-6 bg-muted rounded-full flex items-center justify-center">
                <IconUser className="h-3 w-3 text-muted-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs text-muted-foreground mb-1">
                  <span className="font-medium">{speakerInfo.name}</span>
                  {firstTurn?.turnType && (
                    <span className="ml-2 px-1.5 py-0.5 bg-muted rounded text-xs">
                      {firstTurn.turnType}
                    </span>
                  )}
                </div>
                <p className="text-sm text-foreground/80 line-clamp-2 leading-relaxed">
                  {contentPreview}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Classification Scores */}
        {(humanScore > 0 || personalScore > 0 || relevanceScore > 0) && (
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1">
              <IconBrain className="h-3 w-3 text-blue-500" />
              <span className="text-muted-foreground">Human:</span>
              <span className={cn(
                "font-medium",
                humanScore >= 0.8 ? "text-green-600" :
                humanScore >= 0.6 ? "text-yellow-600" : "text-red-600"
              )}>
                {(humanScore * 100).toFixed(0)}%
              </span>
            </div>
            
            <div className="flex items-center gap-1">
              <IconHeart className="h-3 w-3 text-red-500" />
              <span className="text-muted-foreground">Personal:</span>
              <span className={cn(
                "font-medium",
                personalScore >= 0.7 ? "text-green-600" :
                personalScore >= 0.4 ? "text-yellow-600" : "text-gray-600"
              )}>
                {(personalScore * 100).toFixed(0)}%
              </span>
            </div>
            
            <div className="flex items-center gap-1">
              <IconTarget className="h-3 w-3 text-purple-500" />
              <span className="text-muted-foreground">Relevance:</span>
              <span className={cn(
                "font-medium",
                relevanceScore >= 0.8 ? "text-green-600" :
                relevanceScore >= 0.6 ? "text-yellow-600" : "text-gray-600"
              )}>
                {(relevanceScore * 100).toFixed(0)}%
              </span>
            </div>

            {threadingConfidence && (
              <div className="flex items-center gap-1">
                <IconTrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-muted-foreground">Threading:</span>
                <span className="font-medium text-green-600">
                  {(threadingConfidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}