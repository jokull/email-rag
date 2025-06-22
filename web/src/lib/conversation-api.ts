/**
 * Conversation API Service
 * Handles enhanced conversation queries with similarity search
 */

import { z } from './zero'

export interface SimilarConversation {
  conversation_id: string
  similarity_score: number
  shared_participants: string[]
  topic_overlap: number
  conversation_type: string
  last_message_date: string
  message_count: number
}

export interface ConversationWithSimilarity {
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
  
  turns?: any[]
  elements?: any[]
  similar_conversations?: SimilarConversation[]
}

/**
 * Find conversations similar to the given conversation using multiple criteria
 */
export async function findSimilarConversations(
  conversationId: string,
  participants: string[],
  limit: number = 5
): Promise<SimilarConversation[]> {
  try {
    // Get all conversations for comparison
    const allConversations = await z.query.conversations.run()
    
    // Get conversation turns for topic analysis
    const conversationTurns = await z.query.conversation_turns
      .where('conversation_id', conversationId)
      .run()
    
    // Extract keywords from current conversation for topic matching
    const currentTopics = extractTopicsFromTurns(conversationTurns)
    
    const similarities = await Promise.all(
      allConversations
        .filter(conv => conv.id !== conversationId)
        .map(async (conv) => {
          // Calculate participant similarity
          const sharedParticipants = conv.participants.filter(p => participants.includes(p))
          const participantSimilarity = sharedParticipants.length / 
            Math.max(conv.participants.length, participants.length)
          
          // Calculate topic similarity (simplified - would use embeddings in production)
          const convTurns = await z.query.conversation_turns
            .where('conversation_id', conv.id)
            .run()
          
          const convTopics = extractTopicsFromTurns(convTurns)
          const topicSimilarity = calculateTopicSimilarity(currentTopics, convTopics)
          
          // Get classification for type similarity
          const classification = await z.query.classifications
            .where('thread_id', conv.thread_id)
            .run()
          
          // Combined similarity score
          const similarity = (
            participantSimilarity * 0.4 +      // 40% participant overlap
            topicSimilarity * 0.4 +           // 40% topic similarity  
            (classification?.[0]?.human_score || 0) * 0.2  // 20% human score boost
          )
          
          return {
            conversation_id: conv.id,
            similarity_score: similarity,
            shared_participants: sharedParticipants,
            topic_overlap: topicSimilarity,
            conversation_type: conv.conversation_type,
            last_message_date: conv.last_message_date,
            message_count: conv.message_count
          }
        })
    )
    
    return similarities
      .filter(sim => sim.similarity_score > 0.2)
      .sort((a, b) => b.similarity_score - a.similarity_score)
      .slice(0, limit)
    
  } catch (error) {
    console.error('Error finding similar conversations:', error)
    return []
  }
}

/**
 * Extract topics/keywords from conversation turns
 */
function extractTopicsFromTurns(turns: any[]): string[] {
  if (!turns || turns.length === 0) return []
  
  const allText = turns
    .map(turn => turn.clean_content || '')
    .join(' ')
    .toLowerCase()
  
  // Simple keyword extraction (in production would use proper NLP)
  const words = allText
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 3)
    .filter(word => !isStopWord(word))
  
  // Get word frequency
  const wordCounts = words.reduce((acc, word) => {
    acc[word] = (acc[word] || 0) + 1
    return acc
  }, {} as Record<string, number>)
  
  // Return top 10 most frequent words as topics
  return Object.entries(wordCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([word]) => word)
}

/**
 * Calculate topic similarity between two topic arrays
 */
function calculateTopicSimilarity(topics1: string[], topics2: string[]): number {
  if (topics1.length === 0 || topics2.length === 0) return 0
  
  const set1 = new Set(topics1)
  const set2 = new Set(topics2)
  const intersection = new Set([...set1].filter(x => set2.has(x)))
  const union = new Set([...set1, ...set2])
  
  return intersection.size / union.size // Jaccard similarity
}

/**
 * Simple stop word list
 */
function isStopWord(word: string): boolean {
  const stopWords = new Set([
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'among', 'within', 'without',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just', 'should', 'now', 'would', 'could',
    'there', 'here', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'was', 'were', 'is', 'are', 'am', 'be', 'being'
  ])
  
  return stopWords.has(word)
}

/**
 * Search conversations by text content and filters
 */
export async function searchConversations(
  query: string,
  filters: {
    classification?: string
    importance?: string
    participants?: string[]
    dateRange?: { start: Date; end: Date }
  } = {}
): Promise<ConversationWithSimilarity[]> {
  try {
    // Get all conversations
    let conversations = await z.query.conversations.run()
    
    // Apply filters
    if (filters.dateRange) {
      conversations = conversations.filter(conv => {
        const date = new Date(conv.last_message_date)
        return date >= filters.dateRange!.start && date <= filters.dateRange!.end
      })
    }
    
    if (filters.participants && filters.participants.length > 0) {
      conversations = conversations.filter(conv =>
        filters.participants!.some(p => conv.participants.includes(p))
      )
    }
    
    // Enrich with classification and search content
    const enrichedConversations = await Promise.all(
      conversations.map(async (conv) => {
        // Get classification
        const classification = await z.query.classifications
          .where('thread_id', conv.thread_id)
          .run()
        
        const classificationData = classification?.[0]
        
        // Apply classification filter
        if (filters.classification && 
            classificationData?.classification !== filters.classification) {
          return null
        }
        
        // Apply importance filter
        if (filters.importance === 'high' && 
            (!classificationData || classificationData.importance_score < 0.7)) {
          return null
        }
        
        // Get turns for text search
        const turns = await z.query.conversation_turns
          .where('conversation_id', conv.id)
          .run()
        
        // Text search in content
        if (query) {
          const queryLower = query.toLowerCase()
          const participantMatch = conv.participants.some(p => 
            p.toLowerCase().includes(queryLower)
          )
          const contentMatch = turns.some(turn => 
            turn.clean_content?.toLowerCase().includes(queryLower)
          )
          
          if (!participantMatch && !contentMatch) {
            return null
          }
        }
        
        return {
          ...conv,
          classification: classificationData,
          turns
        }
      })
    )
    
    // Filter out nulls and add similarity data
    const validConversations = enrichedConversations.filter(Boolean) as ConversationWithSimilarity[]
    
    // Add similar conversations for each
    for (const conv of validConversations) {
      conv.similar_conversations = await findSimilarConversations(
        conv.id, 
        conv.participants, 
        3
      )
    }
    
    return validConversations
    
  } catch (error) {
    console.error('Error searching conversations:', error)
    return []
  }
}

/**
 * Get conversation with full detail and similar conversations
 */
export async function getConversationDetail(conversationId: string): Promise<ConversationWithSimilarity | null> {
  try {
    // Get conversation
    const conversations = await z.query.conversations
      .where('id', conversationId)
      .run()
    
    if (!conversations || conversations.length === 0) {
      return null
    }
    
    const conversation = conversations[0]
    
    // Get classification
    const classifications = await z.query.classifications
      .where('thread_id', conversation.thread_id)
      .run()
    
    // Get turns
    const turns = await z.query.conversation_turns
      .where('conversation_id', conversation.id)
      .orderBy('turn_index', 'asc')
      .run()
    
    // Get email elements for enhanced display
    const emailIds = turns.map(turn => turn.email_id).filter(Boolean)
    const elements = emailIds.length > 0 ? await z.query.email_elements
      .where('email_id', 'in', emailIds)
      .orderBy('sequence_number', 'asc')
      .run() : []
    
    // Get similar conversations
    const similarConversations = await findSimilarConversations(
      conversation.id,
      conversation.participants,
      8
    )
    
    return {
      ...conversation,
      classification: classifications?.[0],
      turns,
      elements,
      similar_conversations: similarConversations
    }
    
  } catch (error) {
    console.error('Error getting conversation detail:', error)
    return null
  }
}