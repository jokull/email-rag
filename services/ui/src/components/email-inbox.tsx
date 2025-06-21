import { useState } from 'react'
import { zero } from '../lib/zero'
import { formatDistanceToNow } from 'date-fns'
import { cn } from '../lib/utils'
import { IconMail, IconUser, IconSettings } from 'justd-icons'
import { ThreadActions } from './thread-actions'

export function EmailInbox() {
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null)
  const [showThreadActions, setShowThreadActions] = useState(false)
  
  // Query threads with human classification and recent activity
  const threads = zero.query.threads
    .orderBy('lastMessageDate', 'desc')
    .limit(50)
    .useQuery()

  // Query all classifications to filter human threads
  const allClassifications = zero.query.classifications
    .where('classification', 'human')
    .useQuery()

  // Filter threads that have human classification
  const humanThreads = threads.filter(thread => {
    return allClassifications.some(classification => classification.threadId === thread.id)
  })

  // Query emails for selected thread
  const emailsQuery = selectedThreadId 
    ? zero.query.emails
        .where('threadId', selectedThreadId)
        .orderBy('dateSent', 'asc')
    : null

  const emails = emailsQuery?.useQuery() || []

  // Query classification for selected thread
  const selectedThreadClassification = selectedThreadId
    ? zero.query.classifications
        .where('threadId', selectedThreadId)
        .useQuery()
    : []

  const classification = selectedThreadClassification[0]
  const selectedThread = humanThreads.find(t => t.id === selectedThreadId)

  if (humanThreads.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <div className="text-center">
          <IconMail className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No human conversations found yet.</p>
          <p className="text-sm">Email sync and AI classification may still be processing.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-200px)]">
      {/* Thread List */}
      <div className="lg:col-span-1 border rounded-lg overflow-hidden">
        <div className="bg-muted/50 p-3 border-b">
          <h2 className="font-semibold">Conversations ({humanThreads.length})</h2>
        </div>
        <div className="overflow-y-auto h-full">
          {humanThreads.map((thread) => {
            const participants = Array.isArray(thread.participants) 
              ? thread.participants 
              : []
            const lastMessageDate = new Date(thread.lastMessageDate)
            
            return (
              <div
                key={thread.id}
                className={cn(
                  "p-3 border-b cursor-pointer hover:bg-muted/50 transition-colors",
                  selectedThreadId === thread.id && "bg-muted"
                )}
                onClick={() => setSelectedThreadId(thread.id)}
              >
                <div className="flex items-start justify-between mb-1">
                  <h3 className="font-medium truncate pr-2">
                    {thread.subjectNormalized || 'No Subject'}
                  </h3>
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {formatDistanceToNow(lastMessageDate, { addSuffix: true })}
                  </span>
                </div>
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <IconUser className="h-3 w-3" />
                  <span className="truncate">
                    {participants.length > 0 
                      ? participants.slice(0, 2).join(', ') + 
                        (participants.length > 2 ? ` +${participants.length - 2}` : '')
                      : 'Unknown participants'
                    }
                  </span>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {thread.messageCount} message{thread.messageCount !== 1 ? 's' : ''}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Email Detail */}
      <div className="lg:col-span-2 border rounded-lg overflow-hidden">
        {selectedThreadId ? (
          <div className="h-full flex flex-col">
            <div className="bg-muted/50 p-3 border-b flex justify-between items-center">
              <h2 className="font-semibold">Thread Details</h2>
              <button
                onClick={() => setShowThreadActions(!showThreadActions)}
                className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-foreground"
              >
                <IconSettings className="h-4 w-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {emails.map((email) => {
                const sentDate = new Date(email.dateSent)
                
                return (
                  <div key={email.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="font-medium">
                          {email.fromName || email.fromEmail}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {email.fromEmail}
                        </div>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {sentDate.toLocaleDateString()} {sentDate.toLocaleTimeString()}
                      </div>
                    </div>
                    
                    {email.subject && (
                      <div className="font-medium mb-3 text-sm">
                        Subject: {email.subject}
                      </div>
                    )}
                    
                    <div className="prose prose-sm max-w-none">
                      {email.bodyText ? (
                        <pre className="whitespace-pre-wrap font-sans text-sm">
                          {email.bodyText.slice(0, 1000)}
                          {email.bodyText.length > 1000 && '...'}
                        </pre>
                      ) : (
                        <div className="text-muted-foreground italic">
                          No text content available
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <IconMail className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Select a conversation to view details</p>
            </div>
          </div>
        )}
      </div>

      {/* Thread Actions Panel */}
      {selectedThreadId && showThreadActions && (
        <div className="lg:col-span-1 border rounded-lg overflow-hidden">
          <div className="bg-muted/50 p-3 border-b">
            <h2 className="font-semibold">Thread Actions</h2>
          </div>
          <div className="p-4 overflow-y-auto h-full">
            <ThreadActions
              threadId={selectedThreadId}
              fromEmail={emails[0]?.fromEmail || ''}
              classification={classification ? {
                classification: classification.classification,
                humanScore: classification.humanScore || 0,
                personalScore: classification.personalScore || 0,
                relevanceScore: classification.relevanceScore || 0,
                shouldProcess: classification.shouldProcess || false,
              } : undefined}
              onActionComplete={() => {
                // Refresh data or show success message
                console.log('Action completed')
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}