import { Zero } from "@rocicorp/zero"

export const zero = new Zero({
  server: process.env.NODE_ENV === 'production' 
    ? 'https://your-zero-server.com'
    : 'http://localhost:3000',
  schema: {
    version: 1,
    tables: {
      emails: {
        tableName: 'emails',
        columns: {
          id: { type: 'string' },
          messageId: { type: 'string' },
          threadId: { type: 'string', optional: true },
          fromEmail: { type: 'string' },
          fromName: { type: 'string', optional: true },
          toEmails: { type: 'json' },
          subject: { type: 'string', optional: true },
          bodyText: { type: 'string', optional: true },
          dateSent: { type: 'string' },
          dateReceived: { type: 'string' },
          isRead: { type: 'boolean' },
          createdAt: { type: 'string' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
      threads: {
        tableName: 'threads',
        columns: {
          id: { type: 'string' },
          subjectNormalized: { type: 'string' },
          participants: { type: 'json' },
          firstMessageDate: { type: 'string' },
          lastMessageDate: { type: 'string' },
          messageCount: { type: 'number' },
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
        },
      },
      contacts: {
        tableName: 'contacts',
        columns: {
          id: { type: 'string' },
          email: { type: 'string' },
          name: { type: 'string', optional: true },
          frequencyScore: { type: 'number' },
          relationshipStrength: { type: 'number' },
          totalMessages: { type: 'number' },
        },
        primaryKey: 'id',
      },
      classifications: {
        tableName: 'classifications',
        columns: {
          id: { type: 'string' },
          threadId: { type: 'string' },
          classification: { type: 'string' },
          confidence: { type: 'number' },
        },
        primaryKey: 'id',
        relationships: {
          thread: {
            source: 'threadId',
            dest: { schema: 'threads', field: 'id' },
          },
        },
      },
    },
  },
})