import React, { useEffect, useRef, useState, Component } from 'react'
import Sidebar from './components/Sidebar.jsx'
import RawDataSection from './components/RawDataSection.jsx'
import CleaningSection from './components/CleaningSection.jsx'
import XGBoostSection from './components/XGBoostSection.jsx'
import CalibrationSection from './components/CalibrationSection.jsx'
import BlendSection from './components/BlendSection.jsx'
import ResultsSection from './components/ResultsSection.jsx'
import resultsRaw from './data/results.json?raw'

// ── Parse data at module level, capture any error ──────────────────────────
let resultsData = null
let parseError  = null
try {
  resultsData = JSON.parse(resultsRaw)
} catch (e) {
  parseError = e.message
}

// ── Error boundary: catches render-time crashes in any child ───────────────
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(e) { return { error: e } }
  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center p-8">
          <div className="card max-w-2xl w-full">
            <p className="text-red-400 font-bold text-lg mb-2">Render error</p>
            <p className="text-slate-300 text-sm mb-4">{this.state.error.message}</p>
            <pre className="code-block text-xs whitespace-pre-wrap text-red-300">
              {this.state.error.stack}
            </pre>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

const SECTION_IDS = ['raw-data', 'cleaning', 'xgb-training', 'calibration', 'blend', 'results']

export default function App() {
  const [active, setActive] = useState('raw-data')
  const observerRef = useRef(null)

  useEffect(() => {
    if (!resultsData) return
    observerRef.current?.disconnect()
    const obs = new IntersectionObserver(
      entries => { entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id) }) },
      { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
    )
    SECTION_IDS.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el) })
    observerRef.current = obs
    return () => obs.disconnect()
  }, [])

  // Show parse errors
  if (parseError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <div className="card max-w-xl w-full text-center">
          <p className="text-red-400 font-bold text-lg mb-2">Failed to parse results.json</p>
          <p className="text-slate-300 text-sm">{parseError}</p>
        </div>
      </div>
    )
  }

  const data = resultsData

  return (
    <ErrorBoundary>
      <div className="flex min-h-screen">
        <Sidebar active={active} />

        <main className="flex-1 ml-64 min-h-screen">
          <header className="sticky top-0 z-20 bg-slate-950/90 backdrop-blur-sm
                             border-b border-slate-800 px-8 py-3">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-base font-bold text-white">
                  Customer Churn Prediction Dashboard
                </h1>
                <p className="text-xs text-slate-500">
                  Kaggle Playground Series S6E3 · Generated {
                    data.meta?.generated_at
                      ? new Date(data.meta.generated_at).toLocaleString()
                      : '—'
                  }
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
                  <span><span className="text-white font-semibold">{data.meta?.train_samples?.toLocaleString()}</span> train</span>
                  <span><span className="text-white font-semibold">{data.meta?.test_samples?.toLocaleString()}</span> test</span>
                  <span><span className="text-white font-semibold">{data.meta?.n_features}</span> features</span>
                </div>
                <span className="badge bg-blue-500/20 text-blue-400 border border-blue-500/30 text-xs">
                  4 Models + Blend
                </span>
              </div>
            </div>
          </header>

          <div className="px-8 py-8 max-w-6xl">
            <ErrorBoundary><RawDataSection    data={data} /></ErrorBoundary>
            <ErrorBoundary><CleaningSection   data={data} /></ErrorBoundary>
            <ErrorBoundary><XGBoostSection    data={data} /></ErrorBoundary>
            <ErrorBoundary><CalibrationSection data={data} /></ErrorBoundary>
            <ErrorBoundary><BlendSection      data={data} /></ErrorBoundary>
            <ErrorBoundary><ResultsSection    data={data} /></ErrorBoundary>

            <footer className="mt-8 pt-6 border-t border-slate-800 text-center">
              <p className="text-xs text-slate-600">
                Built with React · Recharts · TailwindCSS · XGBoost · scikit-learn
              </p>
            </footer>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  )
}
