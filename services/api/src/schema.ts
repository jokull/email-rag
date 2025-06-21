import { createSchema, createTableSchema } from '@rocicorp/zero';

const emailSchema = createTableSchema({
  tableName: 'emails',
  columns: {
    id: { type: 'string' },
    messageId: { type: 'string' },
    threadId: { type: 'string', optional: true },
    imapUid: { type: 'number', optional: true },
    mailboxName: { type: 'string', optional: true },
    fromEmail: { type: 'string' },
    fromName: { type: 'string', optional: true },
    toEmails: { type: 'json' },
    ccEmails: { type: 'json', optional: true },
    bccEmails: { type: 'json', optional: true },
    subject: { type: 'string', optional: true },
    bodyText: { type: 'string', optional: true },
    bodyHtml: { type: 'string', optional: true },
    dateSent: { type: 'string' },
    dateReceived: { type: 'string' },
    rawHeaders: { type: 'json', optional: true },
    attachments: { type: 'json', optional: true },
    flags: { type: 'json', optional: true },
    isRead: { type: 'boolean' },
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
});

const threadSchema = createTableSchema({
  tableName: 'threads',
  columns: {
    id: { type: 'string' },
    subjectNormalized: { type: 'string' },
    participants: { type: 'json' },
    firstMessageDate: { type: 'string' },
    lastMessageDate: { type: 'string' },
    messageCount: { type: 'number' },
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
    embeddings: {
      source: 'id',
      dest: { schema: 'embeddings', field: 'threadId' },
    },
  },
});

const contactSchema = createTableSchema({
  tableName: 'contacts',
  columns: {
    id: { type: 'string' },
    email: { type: 'string' },
    name: { type: 'string', optional: true },
    frequencyScore: { type: 'number' },
    relationshipStrength: { type: 'number' },
    firstContact: { type: 'string', optional: true },
    lastContact: { type: 'string', optional: true },
    totalMessages: { type: 'number' },
    createdAt: { type: 'string' },
    updatedAt: { type: 'string' },
  },
  primaryKey: 'id',
});

const classificationSchema = createTableSchema({
  tableName: 'classifications',
  columns: {
    id: { type: 'string' },
    threadId: { type: 'string' },
    classification: { type: 'string' },
    confidence: { type: 'number' },
    modelUsed: { type: 'string' },
    reasoning: { type: 'string', optional: true },
    createdAt: { type: 'string' },
  },
  primaryKey: 'id',
  relationships: {
    thread: {
      source: 'threadId',
      dest: { schema: 'threads', field: 'id' },
    },
  },
});

const embeddingSchema = createTableSchema({
  tableName: 'embeddings',
  columns: {
    id: { type: 'string' },
    threadId: { type: 'string' },
    chunkText: { type: 'string' },
    chunkIndex: { type: 'number' },
    embedding: { type: 'json' },
    createdAt: { type: 'string' },
  },
  primaryKey: 'id',
  relationships: {
    thread: {
      source: 'threadId',
      dest: { schema: 'threads', field: 'id' },
    },
  },
});

const senderRulesSchema = createTableSchema({
  tableName: 'sender_rules',
  columns: {
    id: { type: 'string' },
    emailPattern: { type: 'string' },
    ruleType: { type: 'string' },
    priority: { type: 'number' },
    isActive: { type: 'boolean' },
    createdBy: { type: 'string', optional: true },
    notes: { type: 'string', optional: true },
    createdAt: { type: 'string' },
    updatedAt: { type: 'string' },
  },
  primaryKey: 'id',
});

const userPreferencesSchema = createTableSchema({
  tableName: 'user_preferences',
  columns: {
    id: { type: 'string' },
    preferenceKey: { type: 'string' },
    preferenceValue: { type: 'json' },
    description: { type: 'string', optional: true },
    createdAt: { type: 'string' },
    updatedAt: { type: 'string' },
  },
  primaryKey: 'id',
});

const processingQueueSchema = createTableSchema({
  tableName: 'processing_queue',
  columns: {
    id: { type: 'string' },
    threadId: { type: 'string' },
    queueType: { type: 'string' },
    priority: { type: 'number' },
    status: { type: 'string' },
    attempts: { type: 'number' },
    maxAttempts: { type: 'number' },
    errorMessage: { type: 'string', optional: true },
    processingStartedAt: { type: 'string', optional: true },
    processingCompletedAt: { type: 'string', optional: true },
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
});

const processingStatsSchema = createTableSchema({
  tableName: 'processing_stats',
  columns: {
    id: { type: 'string' },
    date: { type: 'string' },
    statType: { type: 'string' },
    statValue: { type: 'number' },
    metadata: { type: 'json', optional: true },
    createdAt: { type: 'string' },
  },
  primaryKey: 'id',
});

const threadActionsSchema = createTableSchema({
  tableName: 'thread_actions',
  columns: {
    id: { type: 'string' },
    threadId: { type: 'string' },
    actionType: { type: 'string' },
    createdBy: { type: 'string', optional: true },
    notes: { type: 'string', optional: true },
    createdAt: { type: 'string' },
  },
  primaryKey: 'id',
  relationships: {
    thread: {
      source: 'threadId',
      dest: { schema: 'threads', field: 'id' },
    },
  },
});

export const schema = createSchema({
  version: 1,
  tables: {
    emails: emailSchema,
    threads: threadSchema,
    contacts: contactSchema,
    classifications: classificationSchema,
    embeddings: embeddingSchema,
    senderRules: senderRulesSchema,
    userPreferences: userPreferencesSchema,
    processingQueue: processingQueueSchema,
    processingStats: processingStatsSchema,
    threadActions: threadActionsSchema,
  },
});