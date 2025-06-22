import { Link, useLocation } from '@tanstack/react-router'
import { cn } from '../lib/utils'
import { IconMail, IconChart, IconSearch, IconSettings } from 'justd-icons'

export function Navigation() {
  const location = useLocation()
  
  const navItems = [
    {
      href: '/',
      label: 'Inbox',
      icon: <IconMail className="h-4 w-4" />,
      isActive: location.pathname === '/'
    },
    {
      href: '/conversations',
      label: 'Conversations',
      icon: <IconMail className="h-4 w-4" />,
      isActive: location.pathname.startsWith('/conversations')
    },
    {
      href: '/dashboard',
      label: 'Dashboard',
      icon: <IconChart className="h-4 w-4" />,
      isActive: location.pathname === '/dashboard'
    },
    {
      href: '/search',
      label: 'Search',
      icon: <IconSearch className="h-4 w-4" />,
      isActive: location.pathname === '/search'
    }
  ]

  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex space-x-8">
              <div className="flex items-center">
                <h1 className="text-xl font-bold">Email RAG</h1>
              </div>
              
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    to={item.href}
                    className={cn(
                      "inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors",
                      item.isActive
                        ? "border-primary text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300"
                    )}
                  >
                    <span className="flex items-center gap-2">
                      {item.icon}
                      {item.label}
                    </span>
                  </Link>
                ))}
              </div>
            </div>
          </div>
          
          <div className="flex items-center">
            <div className="text-sm text-muted-foreground">
              Real-time sync active
            </div>
          </div>
        </div>
      </div>
      
      {/* Mobile Navigation */}
      <div className="sm:hidden">
        <div className="pt-2 pb-3 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "block pl-3 pr-3 py-2 border-l-4 text-base font-medium transition-colors",
                item.isActive
                  ? "bg-primary/10 border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted hover:border-gray-300"
              )}
            >
              <span className="flex items-center gap-2">
                {item.icon}
                {item.label}
              </span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}