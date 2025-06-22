import { useState } from 'react'
import { zero } from '../lib/zero'
import { cn } from '../lib/utils'
import { IconPlay, IconPause, IconFlag, IconX, IconCheck } from 'justd-icons'

interface ThreadActionsProps {
  threadId: string
  fromEmail: string
  classification?: {
    classification: string
    humanScore: number
    personalScore: number
    relevanceScore: number
    shouldProcess: boolean
  }
  onActionComplete?: () => void
}

export function ThreadActions({ threadId, fromEmail, classification, onActionComplete }: ThreadActionsProps) {
  const [isLoading, setIsLoading] = useState(false)

  // Query existing thread actions
  const threadActions = zero.query.threadActions
    .where('threadId', threadId)
    .useQuery()

  const hasForceProcess = threadActions.some(a => a.actionType === 'force_process')
  const hasSkipProcess = threadActions.some(a => a.actionType === 'skip_processing')
  const hasMarkImportant = threadActions.some(a => a.actionType === 'mark_important')
  const hasMarkSpam = threadActions.some(a => a.actionType === 'mark_spam')

  // Query processing queue status for this thread
  const processingQueue = zero.query.processingQueue
    .where('threadId', threadId)
    .useQuery()

  const classificationStatus = processingQueue.find(q => q.queueType === 'classification')
  const embeddingStatus = processingQueue.find(q => q.queueType === 'embedding')

  const handleAction = async (actionType: string, notes?: string) => {
    setIsLoading(true)
    try {
      // Create thread action record
      await zero.mutate.threadActions.insert({
        id: `action_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        threadId,
        actionType,
        notes: notes || null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      })
      
      onActionComplete?.()
    } catch (error) {
      console.error('Action failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddSenderRule = async (ruleType: 'whitelist' | 'blacklist') => {
    setIsLoading(true)
    try {
      const emailDomain = '@' + fromEmail.split('@')[1]
      
      // Create sender rule
      await zero.mutate.senderRules.insert({
        id: `rule_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        ruleType,
        pattern: emailDomain,
        isActive: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      })
      
      // Also create thread action to track this
      await zero.mutate.threadActions.insert({
        id: `action_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        threadId,
        actionType: `${ruleType}_sender`,
        notes: `Pattern: ${emailDomain}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      })
      
      onActionComplete?.()
    } catch (error) {
      console.error('Sender rule creation failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Classification Status */}
      {classification && (
        <div className="bg-muted/30 rounded-lg p-3">
          <h4 className="font-medium text-sm mb-2">AI Classification</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>Type: <span className="font-medium">{classification.classification}</span></div>
            <div>Human: <span className="font-medium">{(classification.humanScore * 100).toFixed(0)}%</span></div>
            <div>Personal: <span className="font-medium">{(classification.personalScore * 100).toFixed(0)}%</span></div>
            <div>Relevance: <span className="font-medium">{(classification.relevanceScore * 100).toFixed(0)}%</span></div>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className="text-xs">RAG Processing:</span>
            {classification.shouldProcess ? (
              <span className="text-xs text-green-600 font-medium">✓ Eligible</span>
            ) : (
              <span className="text-xs text-red-600 font-medium">✗ Filtered out</span>
            )}
          </div>
        </div>
      )}

      {/* Processing Status */}
      <div className="bg-muted/30 rounded-lg p-3">
        <h4 className="font-medium text-sm mb-2">Processing Status</h4>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span>Classification:</span>
            <span className={cn(
              "font-medium",
              classificationStatus?.status === 'completed' ? "text-green-600" :
              classificationStatus?.status === 'processing' ? "text-blue-600" :
              classificationStatus?.status === 'failed' ? "text-red-600" :
              "text-muted-foreground"
            )}>
              {classificationStatus?.status || 'pending'}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Embeddings:</span>
            <span className={cn(
              "font-medium",
              embeddingStatus?.status === 'completed' ? "text-green-600" :
              embeddingStatus?.status === 'processing' ? "text-blue-600" :
              embeddingStatus?.status === 'failed' ? "text-red-600" :
              "text-muted-foreground"
            )}>
              {embeddingStatus?.status || 'pending'}
            </span>
          </div>
        </div>
      </div>

      {/* Thread Actions */}
      <div className="space-y-2">
        <h4 className="font-medium text-sm">Thread Actions</h4>
        
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => handleAction('force_process')}
            disabled={isLoading || hasForceProcess}
            className={cn(
              "px-3 py-2 text-xs font-medium rounded flex items-center gap-1 transition-colors",
              hasForceProcess 
                ? "bg-green-100 text-green-800 border border-green-200" 
                : "bg-green-500 text-white hover:bg-green-600 disabled:opacity-50"
            )}
          >
            <IconPlay className="h-3 w-3" />
            {hasForceProcess ? 'Force Queued' : 'Force Process'}
          </button>

          <button
            onClick={() => handleAction('skip_processing')}
            disabled={isLoading || hasSkipProcess}
            className={cn(
              "px-3 py-2 text-xs font-medium rounded flex items-center gap-1 transition-colors",
              hasSkipProcess 
                ? "bg-red-100 text-red-800 border border-red-200" 
                : "bg-red-500 text-white hover:bg-red-600 disabled:opacity-50"
            )}
          >
            <IconPause className="h-3 w-3" />
            {hasSkipProcess ? 'Skip Applied' : 'Skip Process'}
          </button>

          <button
            onClick={() => handleAction('mark_important')}
            disabled={isLoading || hasMarkImportant}
            className={cn(
              "px-3 py-2 text-xs font-medium rounded flex items-center gap-1 transition-colors",
              hasMarkImportant 
                ? "bg-yellow-100 text-yellow-800 border border-yellow-200" 
                : "bg-yellow-500 text-white hover:bg-yellow-600 disabled:opacity-50"
            )}
          >
            <IconFlag className="h-3 w-3" />
            {hasMarkImportant ? 'Important' : 'Mark Important'}
          </button>

          <button
            onClick={() => handleAction('mark_spam')}
            disabled={isLoading || hasMarkSpam}
            className={cn(
              "px-3 py-2 text-xs font-medium rounded flex items-center gap-1 transition-colors",
              hasMarkSpam 
                ? "bg-gray-100 text-gray-800 border border-gray-200" 
                : "bg-gray-500 text-white hover:bg-gray-600 disabled:opacity-50"
            )}
          >
            <IconX className="h-3 w-3" />
            {hasMarkSpam ? 'Marked Spam' : 'Mark Spam'}
          </button>
        </div>
      </div>

      {/* Sender Rules */}
      <div className="space-y-2">
        <h4 className="font-medium text-sm">Sender Rules</h4>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => handleAddSenderRule('whitelist')}
            disabled={isLoading}
            className="px-3 py-2 text-xs font-medium rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50 flex items-center gap-1"
          >
            <IconCheck className="h-3 w-3" />
            Whitelist Domain
          </button>
          <button
            onClick={() => handleAddSenderRule('blacklist')}
            disabled={isLoading}
            className="px-3 py-2 text-xs font-medium rounded bg-red-500 text-white hover:bg-red-600 disabled:opacity-50 flex items-center gap-1"
          >
            <IconX className="h-3 w-3" />
            Blacklist Domain
          </button>
        </div>
        <p className="text-xs text-muted-foreground">
          Rules apply to: {fromEmail.split('@')[1]}
        </p>
      </div>
    </div>
  )
}