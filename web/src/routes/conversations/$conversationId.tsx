import { createFileRoute } from '@tanstack/react-router'
import { ConversationDetail } from '../../components/conversation-detail'

export const Route = createFileRoute('/conversations/$conversationId')({
  component: ConversationDetailPage
})

function ConversationDetailPage() {
  return (
    <div className="h-full">
      <ConversationDetail />
    </div>
  )
}