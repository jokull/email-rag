import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">About Email RAG</h1>
      <div className="space-y-4 max-w-2xl">
        <p>
          Email RAG is an AI-powered email analysis system that helps you search
          and understand your email conversations more effectively.
        </p>
        <div>
          <h2 className="text-xl font-semibold mb-2">Features</h2>
          <ul className="list-disc pl-6 space-y-1">
            <li>Automatic email synchronization from IMAP</li>
            <li>AI classification to filter human conversations</li>
            <li>Semantic search using vector embeddings</li>
            <li>Real-time updates with Zero sync</li>
            <li>Thread-based conversation grouping</li>
            <li>Contact relationship tracking</li>
          </ul>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-2">Technology Stack</h2>
          <ul className="list-disc pl-6 space-y-1">
            <li>Docker containerized architecture</li>
            <li>PostgreSQL with pgvector for embeddings</li>
            <li>Node.js IMAP sync service</li>
            <li>Python AI classification and RAG pipeline</li>
            <li>Rocicorp Zero for real-time sync</li>
            <li>React with TanStack Router</li>
          </ul>
        </div>
      </div>
    </div>
  )
}