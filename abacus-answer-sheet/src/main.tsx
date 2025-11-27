import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import MathWorksheetSolver from './components/SheetAnswers.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* <App /> */}
    <MathWorksheetSolver/>
  </StrictMode>,
)
