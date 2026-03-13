import React, { useState } from 'react'
import { CheckCircle, ChevronDown, ChevronUp, Wrench } from 'lucide-react'

export default function CleaningSection({ data }) {
  const { cleaning_steps = [], meta = {} } = data
  const [expanded, setExpanded] = useState(null)

  return (
    <section id="cleaning" className="mb-16 scroll-mt-6">
      <h2 className="section-title">Cleaning Pipeline</h2>
      <p className="section-sub">
        Step-by-step data transformations applied before model training.
        Final dataset: {meta.train_samples?.toLocaleString()} train /&nbsp;
        {meta.test_samples?.toLocaleString()} test samples · {meta.n_features} features
      </p>

      {/* Pipeline flow */}
      <div className="relative">
        {cleaning_steps.map((step, i) => {
          const isOpen = expanded === i
          return (
            <div key={i} className="flex gap-4 mb-2">
              {/* Connector */}
              <div className="flex flex-col items-center shrink-0">
                <div className="w-8 h-8 rounded-full bg-blue-600/20 border border-blue-500/40
                                flex items-center justify-center text-blue-400 text-sm font-bold">
                  {i + 1}
                </div>
                {i < cleaning_steps.length - 1 && (
                  <div className="w-px flex-1 bg-slate-700/60 min-h-[1rem] mt-1" />
                )}
              </div>

              {/* Card */}
              <div className="flex-1 mb-2">
                <button
                  onClick={() => setExpanded(isOpen ? null : i)}
                  className="w-full card text-left hover:border-blue-500/30 transition-colors group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <CheckCircle size={16} className="text-emerald-400 shrink-0" />
                      <span className="font-semibold text-white text-sm">{step.step}</span>
                    </div>
                    {isOpen
                      ? <ChevronUp size={15} className="text-slate-400" />
                      : <ChevronDown size={15} className="text-slate-400" />
                    }
                  </div>

                  {isOpen && (
                    <div className="mt-4 space-y-3" onClick={e => e.stopPropagation()}>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        {step.description}
                      </p>
                      <div>
                        <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">
                          Code
                        </p>
                        <div className="code-block whitespace-pre-wrap">{step.code}</div>
                      </div>
                    </div>
                  )}
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* Feature list */}
      <div className="card mt-4">
        <div className="flex items-center gap-2 mb-3">
          <Wrench size={15} className="text-blue-400" />
          <span className="text-sm font-semibold text-white">
            Final Feature Set ({meta.n_features} features)
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {(meta.features ?? []).map(f => (
            <span key={f} className="badge bg-slate-700/70 text-slate-300 font-mono text-[11px]">
              {f}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
