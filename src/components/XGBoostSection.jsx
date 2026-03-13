import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, ResponsiveContainer, Cell,
} from 'recharts'
import { BrainCircuit, Sliders } from 'lucide-react'

const COLORS = [
  '#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444',
  '#06b6d4','#6366f1','#84cc16','#f97316','#ec4899',
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-300 mb-1 font-semibold">Round {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(5)}
        </p>
      ))}
    </div>
  )
}

export default function XGBoostSection({ data }) {
  const xgb = data.models?.xgboost ?? {}
  const params         = xgb.params ?? {}
  const trainingLog    = xgb.training_log ?? []
  const featureImport  = (xgb.feature_importance ?? []).slice(0, 15)

  return (
    <section id="xgb-training" className="mb-16 scroll-mt-6">
      <h2 className="section-title">XGBoost Training</h2>
      <p className="section-sub">
        Gradient-boosted tree ensemble — parameters, training logs, and feature importances.
      </p>

      {/* Parameters */}
      <div className="card mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Sliders size={16} className="text-blue-400" />
          <span className="text-sm font-semibold text-white">Hyperparameters</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {Object.entries(params).map(([k, v]) => (
            <div key={k} className="bg-slate-800/60 rounded-lg p-3 border border-slate-700/40">
              <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{k}</p>
              <p className="text-sm font-mono font-semibold text-blue-300">{String(v)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Training log chart */}
      {trainingLog.length > 0 && (
        <div className="card mb-6">
          <div className="flex items-center gap-2 mb-4">
            <BrainCircuit size={16} className="text-blue-400" />
            <span className="text-sm font-semibold text-white">
              Training Log — Log-Loss over Boosting Rounds (full dataset)
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trainingLog} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="round"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Boosting Round', position: 'insideBottom', offset: -3,
                         fill: '#64748b', fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Log-Loss', angle: -90, position: 'insideLeft',
                         fill: '#64748b', fontSize: 11 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                formatter={v => <span style={{ color: '#cbd5e1' }}>{v}</span>}
              />
              <Line
                type="monotone" dataKey="train_logloss" name="Train (full data)"
                stroke="#3b82f6" strokeWidth={2} dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature importance */}
      {featureImport.length > 0 && (
        <div className="card">
          <p className="text-sm font-semibold text-white mb-1">
            Feature Importance (Top {featureImport.length})
          </p>
          <p className="text-xs text-slate-400 mb-4">
            Based on total gain across all splits.
          </p>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart
              data={featureImport}
              layout="vertical"
              margin={{ top: 0, right: 30, left: 90, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                tickFormatter={v => v.toFixed(3)}
              />
              <YAxis
                type="category" dataKey="feature"
                tick={{ fill: '#cbd5e1', fontSize: 10, fontFamily: 'monospace' }}
                width={88}
              />
              <Tooltip
                formatter={(v) => [v.toFixed(5), 'Importance']}
                contentStyle={{
                  background: '#1e293b', border: '1px solid #334155',
                  borderRadius: 8, fontSize: 12,
                }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureImport.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  )
}
