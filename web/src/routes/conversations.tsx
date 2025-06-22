import { createFileRoute } from '@tanstack/react-router'
import { ConversationBrowser } from '../components/conversation-browser'

export const Route = createFileRoute('/conversations')({
  component: ConversationsPage
})

function ConversationsPage() {
  return (
    <div className="h-full">
      <ConversationBrowser />
    </div>
  )
}