import { Zero } from "@rocicorp/zero"

export const zero = new Zero({
  server: import.meta.env.VITE_PUBLIC_SERVER || 'http://localhost:4848',
  schema: {
    version: 1,
    tables: {
      threads: {
        tableName: 'threads',
        columns: {
          id: { type: 'string' },
          subjectNormalized: { type: 'string' },
          participants: { type: 'json' },
          messageCount: { type: 'number' },
          lastMessageDate: { type: 'string' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          emails: {
            source: 'id',
            dest: { schema: 'emails', field: 'threadId' },
          },
          classification: {
            source: 'id',
            dest: { schema: 'classifications', field: 'threadId' },
          },
          threadActions: {
            source: 'id',
            dest: { schema: 'threadActions', field: 'threadId' },
          },
        },
      },
      emails: {
        tableName: 'emails',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          messageId: { type: 'string' },
          fromEmail: { type: 'string' },
          fromName: { type: 'string', optional: true },
          toEmail: { type: 'string' },
          subject: { type: 'string', optional: true },
          bodyText: { type: 'string', optional: true },
          bodyHtml: { type: 'string', optional: true },
          dateSent: { type: 'string' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      classifications: {
        tableName: 'classifications',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          classification: { type: 'string' },
          humanScore: { type: 'number' },
          personalScore: { type: 'number' },
          relevanceScore: { type: 'number' },
          shouldProcess: { type: 'boolean' },
          modelUsed: { type: 'string' },
          tokensUsed: { type: 'number' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      embeddings: {
        tableName: 'embeddings',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          chunkText: { type: 'string' },
          chunkIndex: { type: 'number' },
          embedding: { type: 'json' },
          modelUsed: { type: 'string' },
          tokensUsed: { type: 'number' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      senderRules: {
        tableName: 'senderRules',
        columns: {
          id: { type: 'string' },
          ruleType: { type: 'string' },
          pattern: { type: 'string' },
          isActive: { type: 'boolean' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
      },
      userPreferences: {
        tableName: 'userPreferences',
        columns: {
          id: { type: 'string' },
          preferenceKey: { type: 'string' },
          preferenceValue: { type: 'json' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
      },
      processingQueue: {
        tableName: 'processingQueue',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          queueType: { type: 'string' },
          status: { type: 'string' },
          priority: { type: 'number' },
          retryCount: { type: 'number' },
          lastError: { type: 'string', optional: true },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      processingStats: {
        tableName: 'processingStats',
        columns: {
          id: { type: 'string' },
          date: { type: 'string' },
          statType: { type: 'string' },
          statValue: { type: 'number' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
      },
      threadActions: {
        tableName: 'threadActions',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          actionType: { type: 'string' },
          notes: { type: 'string', optional: true },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      conversations: {
        tableName: 'conversations',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          participants: { type: 'json' },
          messageCount: { type: 'number' },
          conversationType: { type: 'string' },
          threadingConfidence: { type: 'number' },
          firstMessageDate: { type: 'string' },
          lastMessageDate: { type: 'string' },
          durationDays: { type: 'number' },
          temporalPatterns: { type: 'json' },
          conversationFlow: { type: 'json' },
          qualityMetrics: { type: 'json' },
          createdAt: { type: 'string' },
          updatedAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
          turns: {
            source: 'id',
            dest: { schema: 'conversationTurns', field: 'conversationId' },
          },
          classification: {
            source: 'threadId',
            dest: { schema: 'classifications', field: 'threadId' },
          },
        },
      },
      conversationTurns: {
        tableName: 'conversationTurns',
        columns: {
          id: { type: 'string' },
          conversationId: { type: 'string' },
          emailId: { type: 'string' },
          speakerEmail: { type: 'string' },
          speakerName: { type: 'string', optional: true },
          turnIndex: { type: 'number' },
          turnType: { type: 'string' },
          cleanContent: { type: 'string' },
          wordCount: { type: 'number' },
          temporalContext: { type: 'json' },
          speakerMetadata: { type: 'json' },
          contentAnalysis: { type: 'json' },
          createdAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          conversation: {
            source: 'conversationId',
            dest: { schema: 'conversations', field: 'id' },
          },
          email: {
            source: 'emailId',
            dest: { schema: 'emails', field: 'id' },
          },
        },
      },
      cleanedEmails: {
        tableName: 'cleanedEmails',
        columns: {
          id: { type: 'string' },
          emailId: { type: 'string' },
          cleanContent: { type: 'string' },
          signatureRemoved: { type: 'string' },
          quotesRemoved: { type: 'string' },
          originalLength: { type: 'number' },
          cleanedLength: { type: 'number' },
          cleaningConfidence: { type: 'number' },
          cleaningMethod: { type: 'string' },
          contentType: { type: 'string' },
          reductionRatio: { type: 'number' },
          createdAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          email: {
            source: 'emailId',
            dest: { schema: 'emails', field: 'id' },
          },
        },
      },
    },
  },
})