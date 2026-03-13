import React, { useEffect, useState, useRef } from 'react'
import Sidebar from './components/Sidebar.jsx'
import RawDataSection from './components/RawDataSection.jsx'
import CleaningSection from './components/CleaningSection.jsx'
import XGBoostSection from './components/XGBoostSection.jsx'
import CalibrationSection from './components/CalibrationSection.jsx'
import BlendSection from './components/BlendSection.jsx'
import ResultsSection from './components/ResultsSection.jsx'
import { Loader2, AlertCircle } from 'lucide-react'

const SECTION_IDS = ['raw-data', 'cleaning', 'xgb-training', 'calibration', 'blend', 'results']

export default function App() {
  const [data, setData]       = useState(null)
  const [error, setError]     = useState(null)
  const [active, setActive]   = useState('raw-data')
  const observerRef           = useRef(null)

  // Load results.json
  useEffect(() => {
    fetch('/results.json')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json()
      })
      .then(setData)
      .catch(e => setError(e.message))
  }, [])

  // Intersection observer for active nav link
  useEffect(() => {
    if (!data) return
    observerRef.current?.disconnect()
    const obs = new IntersectionObserver(
      entries => {
        entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id) })
      },
      { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
    )
    SECTION_IDS.forEach(id => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    observerRef.current = obs
    return () => obs.disconnect()
  }, [data])

  if (error) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="card text-center max-w-md">
        <AlertCircle size={40} className="text-red-400 mx-auto mb-3" />
        <h2 className="text-lg font-bold text-white mb-2">Failed to load results</h2>
        <p className="text-sm text-slate-400 mb-4">{error}</p>
        <p className="text-xs text-slate-500">
          Run <code className="code-block text-xs py-0.5 px-1.5">python model_pipeline.py</code>
          &nbsp;to generate <code>public/results.json</code>
        </p>
      </div>
    </div>
  )

  if (!data) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <Loader2 size={36} className="text-blue-400 animate-spin mx-auto mb-3" />
        <p className="text-slate-400 text-sm">Loading ML results…</p>
      </div>
    </div>
  )

  return (
    <div className="flex min-h-screen">
      <Sidebar active={active} />

      {/* Main content */}
      <main className="flex-1 ml-64 min-h-screen">
        {/* Top header bar */}
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

        {/* Sections */}
        <div className="px-8 py-8 max-w-6xl">
          <RawDataSection   data={data} />
          <CleaningSection  data={data} />
          <XGBoostSection   data={data} />
          <CalibrationSection data={data} />
          <BlendSection     data={data} />
          <ResultsSection   data={data} />

          <footer className="mt-8 pt-6 border-t border-slate-800 text-center">
            <p className="text-xs text-slate-600">
              Built with React · Recharts · TailwindCSS · XGBoost · scikit-learn
            </p>
          </footer>
        </div>
      </main>
    </div>
  )
}
