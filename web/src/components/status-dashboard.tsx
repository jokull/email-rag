import { useMemo } from 'react'
import { formatDistanceToNow, format } from 'date-fns'
import { cn } from '../lib/utils'
import { zero } from '../lib/zero'
import {
  IconActivity,
  IconDatabase,
  IconBrain,
  IconMail,
  IconUsers,
  IconClock,
  IconTrendingUp,
  IconTarget,
  IconHeart,
  IconAlertTriangle,
  IconCheckCircle,
  IconLoader,
  IconServer,
  IconChart
} from 'justd-icons'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray'
  trend?: {
    value: number
    label: string
    isPositive?: boolean
  }
}

function MetricCard({ title, value, subtitle, icon, color = 'gray', trend }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    gray: 'bg-gray-50 text-gray-700 border-gray-200'
  }

  return (
    <div className={cn(
      "rounded-lg border p-4 transition-colors hover:bg-muted/20",
      colorClasses[color]
    )}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            {icon}
            <span className="text-sm font-medium opacity-80">{title}</span>
          </div>
          <div className="text-2xl font-bold mb-1">{value}</div>
          {subtitle && (
            <div className="text-xs opacity-70">{subtitle}</div>
          )}
        </div>
        {trend && (
          <div className={cn(
            "text-xs font-medium px-2 py-1 rounded-full",
            trend.isPositive ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
          )}>
            {trend.isPositive ? '+' : ''}{trend.value}% {trend.label}
          </div>
        )}
      </div>
    </div>
  )
}

interface ProgressBarProps {
  label: string
  current: number
  total: number
  color?: 'blue' | 'green' | 'yellow' | 'red'
}

function ProgressBar({ label, current, total, color = 'blue' }: ProgressBarProps) {
  const percentage = total > 0 ? (current / total) * 100 : 0
  
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground">
          {current.toLocaleString()} / {total.toLocaleString()} ({percentage.toFixed(1)}%)
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={cn("h-2 rounded-full transition-all duration-500", colorClasses[color])}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  )
}

export function StatusDashboard() {
  // Query all emails for total counts
  const allEmailsQuery = zero.query.emails.useQuery()
  
  // Query processed/unprocessed emails
  const processedEmailsQuery = zero.query.emails
    .where('processed', true)
    .useQuery()
  
  // Query threads and conversations
  const threadsQuery = zero.query.threads.useQuery()
  const conversationsQuery = zero.query.conversations
    .related('classification', q => q)
    .related('turns', q => q)
    .useQuery()
  
  // Query classifications
  const classificationsQuery = zero.query.classifications.useQuery()
  
  // Query embeddings for RAG pipeline
  const embeddingsQuery = zero.query.embeddings.useQuery()
  
  // Query cleaned emails
  const cleanedEmailsQuery = zero.query.cleanedEmails.useQuery()

  // Calculate metrics from queries
  const metrics = useMemo(() => {
    const totalEmails = allEmailsQuery.length
    const processedEmails = processedEmailsQuery.length
    const unprocessedEmails = totalEmails - processedEmails
    
    const totalThreads = threadsQuery.length
    const totalConversations = conversationsQuery.length
    const totalClassifications = classificationsQuery.length
    const totalEmbeddings = embeddingsQuery.length
    const totalCleanedEmails = cleanedEmailsQuery.length
    
    // Classification breakdown
    const humanConversations = conversationsQuery.filter(c => 
      c.classification?.classification === 'human'
    ).length
    
    const promotionalEmails = classificationsQuery.filter(c => 
      c.classification === 'promotional'
    ).length
    
    const transactionalEmails = classificationsQuery.filter(c => 
      c.classification === 'transactional'
    ).length
    
    // Processing rates
    const emailProcessingRate = totalEmails > 0 ? (processedEmails / totalEmails) * 100 : 0
    const threadingRate = totalEmails > 0 ? (totalThreads / totalEmails) * 100 : 0
    const classificationRate = totalEmails > 0 ? (totalClassifications / totalEmails) * 100 : 0
    const embeddingRate = totalEmails > 0 ? (totalEmbeddings / totalEmails) * 100 : 0
    
    // Average scores from classifications
    const avgHumanScore = classificationsQuery.length > 0 ? 
      classificationsQuery.reduce((sum, c) => sum + (c.humanScore || 0), 0) / classificationsQuery.length : 0
    
    const avgPersonalScore = classificationsQuery.length > 0 ? 
      classificationsQuery.reduce((sum, c) => sum + (c.personalScore || 0), 0) / classificationsQuery.length : 0
    
    const avgRelevanceScore = classificationsQuery.length > 0 ? 
      classificationsQuery.reduce((sum, c) => sum + (c.relevanceScore || 0), 0) / classificationsQuery.length : 0
    
    // Latest processing activity
    const latestEmails = allEmailsQuery
      .filter(e => e.receivedDate)
      .sort((a, b) => new Date(b.receivedDate).getTime() - new Date(a.receivedDate).getTime())
      .slice(0, 10)
    
    const latestProcessedEmail = latestEmails.find(e => e.processed)
    const lastProcessedTime = latestProcessedEmail?.receivedDate ? 
      new Date(latestProcessedEmail.receivedDate) : null
    
    return {
      totalEmails,
      processedEmails,
      unprocessedEmails,
      totalThreads,
      totalConversations,
      totalClassifications,
      totalEmbeddings,
      totalCleanedEmails,
      humanConversations,
      promotionalEmails,
      transactionalEmails,
      emailProcessingRate,
      threadingRate,
      classificationRate,
      embeddingRate,
      avgHumanScore,
      avgPersonalScore,
      avgRelevanceScore,
      lastProcessedTime,
      latestEmails
    }
  }, [
    allEmailsQuery,
    processedEmailsQuery,
    threadsQuery,
    conversationsQuery,
    classificationsQuery,
    embeddingsQuery,
    cleanedEmailsQuery
  ])

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">System Status</h1>
          <p className="text-muted-foreground">
            Real-time email processing pipeline dashboard
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <IconActivity className="h-4 w-4 text-green-500" />
          <span>Live Updates</span>
          {metrics.lastProcessedTime && (
            <span>
              • Last processed {formatDistanceToNow(metrics.lastProcessedTime, { addSuffix: true })}
            </span>
          )}
        </div>
      </div>

      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Emails"
          value={metrics.totalEmails.toLocaleString()}
          subtitle={`${metrics.unprocessedEmails} pending processing`}
          icon={<IconMail className="h-4 w-4" />}
          color="blue"
        />
        
        <MetricCard
          title="Conversations"
          value={metrics.totalConversations.toLocaleString()}
          subtitle={`From ${metrics.totalThreads} threads`}
          icon={<IconUsers className="h-4 w-4" />}
          color="green"
        />
        
        <MetricCard
          title="Human Conversations"
          value={metrics.humanConversations.toLocaleString()}
          subtitle={`${((metrics.humanConversations / Math.max(metrics.totalConversations, 1)) * 100).toFixed(1)}% of total`}
          icon={<IconBrain className="h-4 w-4" />}
          color="purple"
        />
        
        <MetricCard
          title="RAG Embeddings"
          value={metrics.totalEmbeddings.toLocaleString()}
          subtitle={`${metrics.embeddingRate.toFixed(1)}% coverage`}
          icon={<IconTarget className="h-4 w-4" />}
          color="yellow"
        />
      </div>

      {/* Processing Pipeline Status */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <IconServer className="h-5 w-5" />
          Processing Pipeline Status
        </h2>
        
        <div className="space-y-4">
          <ProgressBar
            label="Email Processing"
            current={metrics.processedEmails}
            total={metrics.totalEmails}
            color="blue"
          />
          
          <ProgressBar
            label="Thread Detection"
            current={metrics.totalThreads}
            total={metrics.totalEmails}
            color="green"
          />
          
          <ProgressBar
            label="Content Cleaning"
            current={metrics.totalCleanedEmails}
            total={metrics.totalEmails}
            color="yellow"
          />
          
          <ProgressBar
            label="AI Classification"
            current={metrics.totalClassifications}
            total={metrics.totalEmails}
            color="purple"
          />
          
          <ProgressBar
            label="Vector Embeddings"
            current={metrics.totalEmbeddings}
            total={metrics.totalEmails}
            color="red"
          />
        </div>
      </div>

      {/* Classification Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <IconChart className="h-5 w-5" />
            Classification Breakdown
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full" />
                <span className="text-sm">Human Conversations</span>
              </div>
              <span className="font-medium">{metrics.humanConversations}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full" />
                <span className="text-sm">Promotional</span>
              </div>
              <span className="font-medium">{metrics.promotionalEmails}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-sm">Transactional</span>
              </div>
              <span className="font-medium">{metrics.transactionalEmails}</span>
            </div>
          </div>
        </div>

        <div className="bg-card rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <IconBrain className="h-5 w-5" />
            AI Classification Scores
          </h2>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <IconBrain className="h-3 w-3 text-blue-500" />
                  Average Human Score
                </span>
                <span className="font-medium">{(metrics.avgHumanScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 bg-blue-500 rounded-full"
                  style={{ width: `${metrics.avgHumanScore * 100}%` }}
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <IconHeart className="h-3 w-3 text-red-500" />
                  Average Personal Score
                </span>
                <span className="font-medium">{(metrics.avgPersonalScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 bg-red-500 rounded-full"
                  style={{ width: `${metrics.avgPersonalScore * 100}%` }}
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <IconTarget className="h-3 w-3 text-purple-500" />
                  Average Relevance Score
                </span>
                <span className="font-medium">{(metrics.avgRelevanceScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 bg-purple-500 rounded-full"
                  style={{ width: `${metrics.avgRelevanceScore * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <IconClock className="h-5 w-5" />
          Recent Email Activity
        </h2>
        
        <div className="space-y-3">
          {metrics.latestEmails.slice(0, 5).map((email, index) => {
            const receivedTime = new Date(email.receivedDate)
            return (
              <div key={email.id || index} className="flex items-center justify-between py-2 border-b last:border-b-0">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "w-2 h-2 rounded-full",
                    email.processed ? "bg-green-500" : "bg-yellow-500"
                  )} />
                  <div className="text-sm">
                    <div className="font-medium truncate max-w-xs">
                      {email.subject || 'No subject'}
                    </div>
                    <div className="text-muted-foreground text-xs">
                      From: {email.sender || 'Unknown sender'}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground">
                  {formatDistanceToNow(receivedTime, { addSuffix: true })}
                </div>
              </div>
            )
          })}
          
          {metrics.latestEmails.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <IconMail className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No recent email activity</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}