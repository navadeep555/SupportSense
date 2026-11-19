// App shell: top bar with navigation + backend health, and the page routes.
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { api } from './api/client'
import { useAsync } from './api/useAsync'
import { NewTicketPage } from './pages/NewTicketPage'
import { QueuePage } from './pages/QueuePage'
import { ReportsPage } from './pages/ReportsPage'
import { TicketPage } from './pages/TicketPage'

function HealthIndicator() {
  const { data, error } = useAsync(() => api.health(), [])
  if (error) return <span className="health bad" title={error}>● API offline</span>
  if (!data) return null
  const problems = [
    !data.rag_index_loaded && 'RAG index not loaded',
    !data.llm.available && `LLM (${data.llm.model}) not available`,
    !data.classifier_loaded && 'classifier not loaded',
  ].filter(Boolean)
  return (
    <span className={`health ${problems.length ? 'warn' : 'ok'}`}
          title={problems.length ? problems.join('; ') : 'All services ready'}>
      ● {data.llm.provider}: {data.llm.model}
    </span>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <h1>SupportSense</h1>
        <nav>
          <NavLink to="/" end>Queue</NavLink>
          <NavLink to="/new">New ticket</NavLink>
          <NavLink to="/reports">Reports</NavLink>
        </nav>
        <HealthIndicator />
      </header>
      <main>
        <Routes>
          <Route path="/" element={<QueuePage />} />
          <Route path="/new" element={<NewTicketPage />} />
          <Route path="/tickets/:id" element={<TicketPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
