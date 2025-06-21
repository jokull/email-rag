import { useState, useEffect } from 'react'
import { zero } from '../lib/zero'
import { cn } from '../lib/utils'
import { IconLoader, IconCheck, IconX, IconPlay, IconPause } from 'justd-icons'

interface ProcessingStats {
  totalThreads: number
  classifiedThreads: number
  embeddedThreads: number
  pendingClassification: number
  pendingEmbedding: number
  dailyClassificationTokens: number
  dailyEmbeddingTokens: number
  classificationBudget: number
  embeddingBudget: number
}

export function ProcessingDashboard() {
  const [stats, setStats] = useState<ProcessingStats | null>(null)
  const [refreshInterval, setRefreshInterval] = useState(30000) // 30 seconds
  const [isLoading, setIsLoading] = useState(false)

  // Query processing queue for real-time updates
  const processingQueue = zero.query.processingQueue
    .where('status', 'processing')
    .useQuery()

  const pendingClassification = zero.query.processingQueue
    .where('queueType', 'classification')
    .where('status', 'pending')
    .limit(100)
    .useQuery()

  const pendingEmbedding = zero.query.processingQueue
    .where('queueType', 'embedding')
    .where('status', 'pending')
    .limit(100)
    .useQuery()

  // Query daily stats
  const todayStats = zero.query.processingStats
    .where('date', new Date().toISOString().split('T')[0])
    .useQuery()

  // Query user preferences for budgets
  const preferences = zero.query.userPreferences.useQuery()

  useEffect(() => {
    const calculateStats = () => {
      const budgetPref = preferences.find(p => p.preferenceKey === 'daily_processing_budget')
      const budget = budgetPref?.preferenceValue || { classification_tokens: 100000, embedding_tokens: 50000 }

      const classificationTokens = todayStats.find(s => s.statType === 'classification_tokens')?.statValue || 0
      const embeddingTokens = todayStats.find(s => s.statType === 'embedding_tokens')?.statValue || 0

      setStats({
        totalThreads: 0, // Would need a separate query
        classifiedThreads: 0, // Would need a separate query
        embeddedThreads: 0, // Would need a separate query
        pendingClassification: pendingClassification.length,
        pendingEmbedding: pendingEmbedding.length,
        dailyClassificationTokens: classificationTokens,
        dailyEmbeddingTokens: embeddingTokens,
        classificationBudget: budget.classification_tokens,
        embeddingBudget: budget.embedding_tokens,
      })
    }

    calculateStats()
  }, [pendingClassification, pendingEmbedding, todayStats, preferences])

  const currentlyProcessing = processingQueue.filter(item => item.status === 'processing')

  const handleProcessingControl = async (action: 'resume' | 'pause') => {
    setIsLoading(true)
    try {
      // Update user preferences to control processing
      const prefId = `pref_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      await zero.mutate.userPreferences.insert({
        id: prefId,
        preferenceKey: 'processing_enabled',
        preferenceValue: action === 'resume' ? true : false,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      })
    } catch (error) {
      console.error('Processing control failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center p-8">
        <IconLoader className="h-6 w-6 animate-spin" />
      </div>
    )
  }

  const classificationProgress = (stats.dailyClassificationTokens / stats.classificationBudget) * 100
  const embeddingProgress = (stats.dailyEmbeddingTokens / stats.embeddingBudget) * 100

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Processing Queue Status */}
        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">Classification Queue</h3>
          <div className="text-2xl font-bold">{stats.pendingClassification}</div>
          <p className="text-xs text-muted-foreground">threads pending</p>
        </div>

        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">Embedding Queue</h3>
          <div className="text-2xl font-bold">{stats.pendingEmbedding}</div>
          <p className="text-xs text-muted-foreground">threads pending</p>
        </div>

        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">Currently Processing</h3>
          <div className="text-2xl font-bold">{currentlyProcessing.length}</div>
          <p className="text-xs text-muted-foreground">active jobs</p>
        </div>

        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">Total Queue</h3>
          <div className="text-2xl font-bold">{stats.pendingClassification + stats.pendingEmbedding}</div>
          <p className="text-xs text-muted-foreground">items waiting</p>
        </div>
      </div>

      {/* Budget Progress */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold mb-3">Daily Classification Budget</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Used: {stats.dailyClassificationTokens.toLocaleString()} tokens</span>
              <span>{classificationProgress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className={cn(
                  "h-2 rounded-full transition-all",
                  classificationProgress < 80 ? "bg-green-500" : 
                  classificationProgress < 95 ? "bg-yellow-500" : "bg-red-500"
                )}
                style={{ width: `${Math.min(100, classificationProgress)}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              Budget: {stats.classificationBudget.toLocaleString()} tokens
            </div>
          </div>
        </div>

        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold mb-3">Daily Embedding Budget</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Used: {stats.dailyEmbeddingTokens.toLocaleString()} tokens</span>
              <span>{embeddingProgress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className={cn(
                  "h-2 rounded-full transition-all",
                  embeddingProgress < 80 ? "bg-blue-500" : 
                  embeddingProgress < 95 ? "bg-yellow-500" : "bg-red-500"
                )}
                style={{ width: `${Math.min(100, embeddingProgress)}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              Budget: {stats.embeddingBudget.toLocaleString()} tokens
            </div>
          </div>
        </div>
      </div>

      {/* Currently Processing Items */}
      {currentlyProcessing.length > 0 && (
        <div className="bg-card border rounded-lg p-4">
          <h3 className="font-semibold mb-3">Currently Processing</h3>
          <div className="space-y-2">
            {currentlyProcessing.map((item) => (
              <div key={item.id} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <div className="flex items-center gap-2">
                  <IconLoader className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-medium">
                    {item.queueType === 'classification' ? 'Classifying' : 'Generating embeddings'}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Priority: {item.priority}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Processing Controls */}
      <div className="bg-card border rounded-lg p-4">
        <h3 className="font-semibold mb-3">Processing Controls</h3>
        <div className="flex gap-2">
          <button 
            className="px-3 py-2 bg-green-500 text-white rounded text-sm font-medium hover:bg-green-600 flex items-center gap-2 disabled:opacity-50"
            disabled={isLoading || (classificationProgress >= 100 && embeddingProgress >= 100)}
            onClick={() => handleProcessingControl('resume')}
          >
            <IconPlay className="h-4 w-4" />
            Resume Processing
          </button>
          <button 
            className="px-3 py-2 bg-red-500 text-white rounded text-sm font-medium hover:bg-red-600 flex items-center gap-2 disabled:opacity-50"
            disabled={isLoading}
            onClick={() => handleProcessingControl('pause')}
          >
            <IconPause className="h-4 w-4" />
            Pause Processing
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Processing automatically pauses when daily budgets are exceeded
        </p>
      </div>
    </div>
  )
}