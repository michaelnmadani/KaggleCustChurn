import React from 'react'
import {
  Database, Wrench, BrainCircuit, LineChart, BarChart3, ChevronRight,
} from 'lucide-react'

const NAV_ITEMS = [
  { id: 'raw-data',     label: 'Raw Data Overview',     icon: Database },
  { id: 'cleaning',     label: 'Cleaning Pipeline',      icon: Wrench },
  { id: 'xgb-training', label: 'XGBoost Training',       icon: BrainCircuit },
  { id: 'calibration',  label: 'Model Fit & Calibration', icon: LineChart },
  { id: 'results',      label: 'Results & Testing',       icon: BarChart3 },
]

export default function Sidebar({ active }) {
  const scrollTo = (id) => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-slate-900 border-r border-slate-700/60
                      flex flex-col z-30 shadow-2xl">
      {/* Logo / header */}
      <div className="px-5 py-5 border-b border-slate-700/60">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600
                          flex items-center justify-center text-white font-bold text-sm shrink-0">
            ML
          </div>
          <div>
            <p className="text-white font-semibold text-sm leading-tight">Churn Predictor</p>
            <p className="text-slate-500 text-xs">S6E3 Dashboard</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <p className="px-3 text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">
          Sections
        </p>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = active === id
          return (
            <button
              key={id}
              onClick={() => scrollTo(id)}
              className={`w-full nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
            >
              <Icon size={16} className="shrink-0" />
              <span className="flex-1 text-left">{label}</span>
              {isActive && <ChevronRight size={12} />}
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-700/60">
        <p className="text-xs text-slate-500 leading-relaxed">
          Kaggle Playground<br />
          <span className="text-slate-400">Series 6, Episode 3</span>
        </p>
      </div>
    </aside>
  )
}
