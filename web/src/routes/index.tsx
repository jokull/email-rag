import { createFileRoute } from '@tanstack/react-router'
import { ConversationListPage } from '../components/conversation-list-page'
import { Navigation } from '../components/navigation'

export const Route = createFileRoute('/')({
  component: Index,
})

function Index() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="max-w-7xl mx-auto">
        <ConversationListPage />
      </div>
    </div>
  )
}