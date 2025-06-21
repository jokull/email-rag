import { createFileRoute } from '@tanstack/react-router'
import { EmailInbox } from '../components/email-inbox'

export const Route = createFileRoute('/')({
  component: Index,
})

function Index() {
  return (
    <div className="p-4">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Email RAG Search</h1>
        <p className="text-muted-foreground">
          AI-powered search through your email conversations
        </p>
      </div>
      <EmailInbox />
    </div>
  )
}