import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from 'recharts'
import { BarChart3, Award } from 'lucide-react'

// Base models
const MODEL_KEYS = ['xgboost', 'random_forest', 'logistic_regression', 'linear_regression']
const MODEL_LABELS = {
  xgboost:             'XGBoost',
  random_forest:       'Random Forest',
  logistic_regression: 'Logistic Regression',
  linear_regression:   'Linear Regression',
}
const MODEL_COLORS = {
  xgboost:             '#3b82f6',
  random_forest:       '#10b981',
  logistic_regression: '#f59e0b',
  linear_regression:   '#ec4899',
}

// Blend variants (virtual model keys for the selector)
const BLEND_KEYS   = ['blend_simple', 'blend_auc']
const BLEND_LABELS = { blend_simple: 'Blend (Simple)', blend_auc: 'Blend (AUC-Wtd)' }
const BLEND_COLORS = { blend_simple: '#8b5cf6',        blend_auc: '#06b6d4' }

const ALL_KEYS   = [...MODEL_KEYS, ...BLEND_KEYS]
const ALL_LABELS = { ...MODEL_LABELS, ...BLEND_LABELS }
const ALL_COLORS = { ...MODEL_COLORS, ...BLEND_COLORS }

function ConfusionMatrix({ matrix, label, color }) {
  if (!matrix?.length) return null
  const [[tn, fp], [fn, tp]] = matrix
  const total = tn + fp + fn + tp
  const cells = [
    { label: 'TN', value: tn, desc: 'True Negative',  bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
    { label: 'FP', value: fp, desc: 'False Positive',  bg: 'bg-red-500/20',    text: 'text-red-400' },
    { label: 'FN', value: fn, desc: 'False Negative',  bg: 'bg-orange-500/20', text: 'text-orange-400' },
    { label: 'TP', value: tp, desc: 'True Positive',   bg: 'bg-blue-500/20',   text: 'text-blue-400' },
  ]
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
        <span className="text-sm font-semibold text-white">{label}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 mb-3">
        {cells.map(c => (
          <div key={c.label} className={`${c.bg} rounded-lg p-3 text-center`}>
            <div className={`text-2xl font-bold ${c.text}`}>{c.value.toLocaleString()}</div>
            <div className="text-[10px] text-slate-400 mt-0.5">{c.desc}</div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              {((c.value / total) * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
      <div className="text-center text-xs text-slate-500">
        <span className="text-slate-400 font-semibold">Predicted →</span>
        &nbsp;|&nbsp;
        <span className="text-slate-400 font-semibold">↓ Actual</span>
      </div>
    </div>
  )
}

const MetricTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-200 font-semibold mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.fill }}>
          {p.name}: {(p.value * 100).toFixed(2)}%
        </p>
      ))}
    </div>
  )
}

const BAND_KEY_MAP = {
  xgboost:             'xgboost',
  random_forest:       'random_forest',
  logistic_regression: 'logistic_regression',
  linear_regression:   'linear_regression',
  blend_simple:        'blend_simple',
  blend_auc:           'blend_auc_weighted',
}

export default function ResultsSection({ data }) {
  const { models = {}, blend = {}, model_comparison = [], score_bands = {} } = data
  const [selectedModel, setSelectedModel] = useState('xgboost')

  // Resolve metrics for any key (including blend virtuals)
  const getMetrics = (key) => {
    if (key === 'blend_simple')  return blend.simple_blend?.metrics
    if (key === 'blend_auc')     return blend.auc_weighted_blend?.metrics
    return models[key]?.metrics
  }

  const metricsForChart = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

  // Grouped bar data — all 6 entries
  const barData = metricsForChart.map(m => {
    const row = { metric: m.replace('_', ' ').toUpperCase() }
    ALL_KEYS.forEach(k => { row[ALL_LABELS[k]] = getMetrics(k)?.[m] ?? 0 })
    return row
  })

  // Best by F1 (from comparison table)
  const best = model_comparison.reduce((b, r) =>
    (r.f1 ?? 0) > (b.f1 ?? 0) ? r : b, model_comparison[0] ?? {}
  )

  return (
    <section id="results" className="mb-16 scroll-mt-6">
      <h2 className="section-title">Results &amp; Testing</h2>
      <p className="section-sub">
        Confusion matrices, performance metrics, and full model comparison on the held-out test set
        — including both blended ensemble variants.
      </p>

      {/* Best model banner */}
      {best.model && (
        <div className="flex items-center gap-3 bg-amber-500/10 border border-amber-500/30
                        rounded-xl px-4 py-3 mb-6">
          <Award size={18} className="text-amber-400 shrink-0" />
          <div>
            <span className="text-sm text-amber-300 font-semibold">
              Best F1-Score: {best.model}
            </span>
            <span className="text-xs text-slate-400 ml-2">
              F1={best.f1?.toFixed(4)} · AUC={best.roc_auc?.toFixed(4)} · Acc={best.accuracy?.toFixed(4)}
            </span>
          </div>
        </div>
      )}

      {/* Comparison table */}
      <div className="card mb-6 overflow-x-auto">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 size={15} className="text-blue-400" />
          <span className="text-sm font-semibold text-white">Full Model Comparison Table</span>
        </div>
        <table className="w-full text-sm min-w-[640px]">
          <thead>
            <tr className="border-b border-slate-700">
              {['Model','Accuracy','Precision','Recall','F1','AUC','Log-Loss'].map(h => (
                <th key={h} className="px-3 py-2 text-left text-slate-400 font-semibold text-xs">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {model_comparison.map((row, i) => {
              const isBest  = row.model === best.model
              const isBlend = row.model.startsWith('Blend')
              return (
                <tr
                  key={i}
                  className={`border-b border-slate-800 transition-colors ${
                    isBest  ? 'bg-amber-500/5' :
                    isBlend ? 'bg-purple-500/5' :
                    'hover:bg-slate-800/30'
                  }`}
                >
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      {isBest && <Award size={12} className="text-amber-400" />}
                      {isBlend && !isBest && (
                        <span className="w-2 h-2 rounded-full bg-purple-400 shrink-0" />
                      )}
                      <span className={`font-medium ${
                        isBest ? 'text-amber-300' : isBlend ? 'text-purple-300' : 'text-white'
                      }`}>
                        {row.model}
                      </span>
                    </div>
                  </td>
                  {['accuracy','precision','recall','f1','roc_auc','log_loss'].map(k => (
                    <td key={k} className="px-3 py-2.5 font-mono text-xs">
                      <span className={`
                        ${k === 'f1' && isBest ? 'text-amber-400 font-bold' : ''}
                        ${k === 'log_loss' ? 'text-slate-400' : 'text-slate-200'}
                      `}>
                        {row[k]?.toFixed(4)}
                      </span>
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Grouped bar chart */}
      <div className="card mb-6">
        <p className="text-sm font-semibold text-white mb-1">Performance Metrics — All Models + Blends</p>
        <p className="text-xs text-slate-400 mb-4">Side-by-side comparison across key metrics</p>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <YAxis
              domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }}
              tickFormatter={v => (v * 100).toFixed(0) + '%'}
            />
            <Tooltip content={<MetricTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11 }}
              formatter={v => <span style={{ color: '#cbd5e1' }}>{v}</span>}
            />
            {ALL_KEYS.map(k => (
              <Bar
                key={k}
                dataKey={ALL_LABELS[k]}
                fill={ALL_COLORS[k]}
                radius={[3, 3, 0, 0]}
                opacity={BLEND_KEYS.includes(k) ? 0.85 : 1}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Score bands table */}
      {(() => {
        const bands = score_bands[BAND_KEY_MAP[selectedModel]] ?? []
        const maxChurn = Math.max(...bands.map(b => b.actual_churn_pct ?? 0), 1)
        return (
          <div className="card mb-6">
            <p className="text-sm font-semibold text-white mb-1">Prediction Score Bands</p>
            <p className="text-xs text-slate-400 mb-4">
              OOF predicted probability buckets — customer count and observed churn rate per band.
              Select a model above to update.
            </p>
            <div className="flex flex-wrap gap-2 mb-4">
              {ALL_KEYS.map(k => {
                const isBlend = BLEND_KEYS.includes(k)
                return (
                  <button
                    key={k}
                    onClick={() => setSelectedModel(k)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                      selectedModel === k ? 'text-white' : 'border-slate-600 text-slate-400 hover:text-slate-200'
                    }`}
                    style={selectedModel === k ? {
                      background: ALL_COLORS[k] + '33',
                      borderColor: ALL_COLORS[k] + '88',
                      color: ALL_COLORS[k],
                    } : {}}
                  >
                    {isBlend && <span className="mr-1 opacity-70">◈</span>}
                    {ALL_LABELS[k]}
                  </button>
                )
              })}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    {['Predicted Churn Band', 'Customers', '% of Total', 'Actual Churn %', 'Lift vs Base'].map(h => (
                      <th key={h} className="px-3 py-2 text-left text-slate-400 font-semibold text-xs">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const totalCustomers = bands.reduce((s, b) => s + b.customers, 0)
                    const baseRate = data.dataset_stats?.churn_rate ?? null
                    return bands.map((row, i) => {
                      const pctOfTotal = totalCustomers > 0
                        ? ((row.customers / totalCustomers) * 100).toFixed(1)
                        : '—'
                      const lift = baseRate && row.actual_churn_pct != null
                        ? (row.actual_churn_pct / 100 / baseRate).toFixed(2) + '×'
                        : '—'
                      const churnPct = row.actual_churn_pct
                      const barWidth = churnPct != null
                        ? Math.round((churnPct / maxChurn) * 100)
                        : 0
                      const color = ALL_COLORS[selectedModel]
                      return (
                        <tr
                          key={i}
                          className={`border-b border-slate-800 ${i % 2 === 0 ? 'bg-slate-800/20' : ''}`}
                        >
                          <td className="px-3 py-2.5 font-mono text-xs text-slate-200 font-semibold">
                            {row.band}
                          </td>
                          <td className="px-3 py-2.5 font-mono text-xs text-slate-300 text-right">
                            {row.customers.toLocaleString()}
                          </td>
                          <td className="px-3 py-2.5 font-mono text-xs text-slate-400 text-right">
                            {pctOfTotal}%
                          </td>
                          <td className="px-3 py-2.5">
                            {churnPct != null ? (
                              <div className="flex items-center gap-2">
                                <div className="flex-1 bg-slate-700/50 rounded-full h-1.5 min-w-[60px]">
                                  <div
                                    className="h-1.5 rounded-full transition-all"
                                    style={{ width: `${barWidth}%`, background: color }}
                                  />
                                </div>
                                <span className="font-mono text-xs text-slate-200 w-12 text-right">
                                  {churnPct.toFixed(1)}%
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-500 text-xs">—</span>
                            )}
                          </td>
                          <td className={`px-3 py-2.5 font-mono text-xs text-right ${
                            lift !== '—' && parseFloat(lift) >= 2 ? 'text-emerald-400' :
                            lift !== '—' && parseFloat(lift) >= 1 ? 'text-slate-300' :
                            'text-slate-500'
                          }`}>
                            {lift}
                          </td>
                        </tr>
                      )
                    })
                  })()}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}

      {/* Confusion matrices */}
      <div>
        <p className="text-sm font-semibold text-white mb-1">Confusion Matrices</p>
        <p className="text-xs text-slate-400 mb-4">
          Select any model or blend variant to inspect its confusion matrix on the test set.
        </p>
        <div className="flex flex-wrap gap-2 mb-4">
          {ALL_KEYS.map(k => {
            const isBlend = BLEND_KEYS.includes(k)
            return (
              <button
                key={k}
                onClick={() => setSelectedModel(k)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  selectedModel === k ? 'text-white' : 'border-slate-600 text-slate-400 hover:text-slate-200'
                }`}
                style={selectedModel === k ? {
                  background: ALL_COLORS[k] + '33',
                  borderColor: ALL_COLORS[k] + '88',
                  color: ALL_COLORS[k],
                } : {}}
              >
                {isBlend && <span className="mr-1 opacity-70">◈</span>}
                {ALL_LABELS[k]}
              </button>
            )
          })}
        </div>

        <div className="max-w-xs">
          <ConfusionMatrix
            matrix={getMetrics(selectedModel)?.confusion_matrix}
            label={ALL_LABELS[selectedModel]}
            color={ALL_COLORS[selectedModel]}
          />
        </div>

        {/* Metric cards for selected model */}
        {getMetrics(selectedModel) && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 mt-4">
            {[
              { key: 'accuracy',  label: 'Accuracy',  color: 'text-blue-400' },
              { key: 'precision', label: 'Precision', color: 'text-purple-400' },
              { key: 'recall',    label: 'Recall',    color: 'text-emerald-400' },
              { key: 'f1',        label: 'F1-Score',  color: 'text-amber-400' },
              { key: 'roc_auc',   label: 'AUC-ROC',   color: 'text-cyan-400' },
              { key: 'log_loss',  label: 'Log-Loss',  color: 'text-red-400' },
            ].map(({ key, label, color }) => (
              <div key={key} className="metric-card">
                <div className={`metric-value ${color}`}>
                  {getMetrics(selectedModel)[key]?.toFixed(4)}
                </div>
                <div className="metric-label">{label}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
