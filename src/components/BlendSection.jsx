import React, { useState } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, ResponsiveContainer, Legend,
} from 'recharts'
import { Layers, Scale, TrendingUp, Info } from 'lucide-react'

const MODEL_DISPLAY = {
  xgboost:             'XGBoost',
  random_forest:       'Random Forest',
  logistic_regression: 'Logistic Regression',
  linear_regression:   'Linear Regression',
}

const VARIANT_COLORS = {
  simple:      '#8b5cf6',
  auc_weighted:'#06b6d4',
}

const BASE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899']


function WeightTable({ weights, color }) {
  return (
    <div className="space-y-2">
      {weights.map((w, i) => (
        <div key={w.model} className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-300 w-36 shrink-0">
            {MODEL_DISPLAY[w.model] ?? w.model}
          </span>
          <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${(w.weight * 100).toFixed(1)}%`, background: BASE_COLORS[i] }}
            />
          </div>
          <span className="text-xs font-mono w-12 text-right" style={{ color: BASE_COLORS[i] }}>
            {(w.weight * 100).toFixed(1)}%
          </span>
          <span className="text-xs text-slate-500 w-16 text-right">
            AUC {w.auc.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  )
}

function MetricCards({ metrics, color }) {
  return (
    <div className="grid grid-cols-3 gap-2 mt-4">
      {[
        { key: 'accuracy',  label: 'Accuracy' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall',    label: 'Recall' },
        { key: 'f1',        label: 'F1-Score' },
        { key: 'roc_auc',   label: 'AUC-ROC' },
        { key: 'log_loss',  label: 'Log-Loss' },
      ].map(({ key, label }) => (
        <div key={key} className="bg-slate-800/60 rounded-lg p-2.5 text-center border border-slate-700/40">
          <div className="text-lg font-bold" style={{ color }}>
            {metrics[key]?.toFixed(4)}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mt-0.5">{label}</div>
        </div>
      ))}
    </div>
  )
}

export default function BlendSection({ data }) {
  const blend   = data.blend ?? {}
  const models  = data.models ?? {}
  const simple  = blend.simple_blend ?? {}
  const aucWtd  = blend.auc_weighted_blend ?? {}
  const [activeTab, setActiveTab] = useState('simple')

  const current = activeTab === 'simple' ? simple : aucWtd
  const currentColor = activeTab === 'simple' ? VARIANT_COLORS.simple : VARIANT_COLORS.auc_weighted

  // ROC comparison: both blends vs all base models
  const rocSeries = [
    { key: 'simple',      label: 'Blend (Simple)',      color: VARIANT_COLORS.simple,       data: simple.metrics?.roc_curve },
    { key: 'auc_weighted',label: 'Blend (AUC-Wtd)',     color: VARIANT_COLORS.auc_weighted, data: aucWtd.metrics?.roc_curve },
    ...Object.entries(models).map(([k, v], i) => ({
      key: k, label: MODEL_DISPLAY[k] ?? k, color: BASE_COLORS[i],
      data: v.metrics?.roc_curve,
    })),
  ]
  const maxLen = Math.max(...rocSeries.map(s => s.data?.fpr?.length ?? 0))
  const rocData = Array.from({ length: maxLen }, (_, i) => {
    const row = { fpr: rocSeries[0]?.data?.fpr?.[i] ?? 0 }
    rocSeries.forEach(s => { row[s.key] = s.data?.tpr?.[i] ?? null })
    return row
  })

  // F1 lift bar chart: base models vs blends
  const liftData = [
    ...Object.entries(models).map(([k, v], i) => ({
      name: MODEL_DISPLAY[k] ?? k,
      f1: parseFloat((v.metrics?.f1 ?? 0).toFixed(4)),
      auc: parseFloat((v.metrics?.roc_auc ?? 0).toFixed(4)),
      color: BASE_COLORS[i],
    })),
    { name: 'Blend (Simple)',   f1: parseFloat((simple.metrics?.f1 ?? 0).toFixed(4)),  auc: parseFloat((simple.metrics?.roc_auc ?? 0).toFixed(4)), color: VARIANT_COLORS.simple },
    { name: 'Blend (AUC-Wtd)', f1: parseFloat((aucWtd.metrics?.f1 ?? 0).toFixed(4)),  auc: parseFloat((aucWtd.metrics?.roc_auc ?? 0).toFixed(4)), color: VARIANT_COLORS.auc_weighted },
  ]

  return (
    <section id="blend" className="mb-16 scroll-mt-6">
      <h2 className="section-title">Blended Ensemble</h2>
      <p className="section-sub">
        Soft-voting ensemble — predicted probabilities from all four models are averaged using two
        weighting strategies, then thresholded at 0.5 for the final class decision.
      </p>

      {/* How it works */}
      <div className="card mb-6">
        <div className="flex items-start gap-3">
          <Info size={16} className="text-blue-400 mt-0.5 shrink-0" />
          <div className="text-sm text-slate-300 leading-relaxed">
            <span className="text-white font-semibold">How the blend works: </span>
            {blend.description}
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-400">
              <span>
                <span className="font-mono text-purple-400">Simple avg:</span>
                &nbsp;ŷ = (p_xgb + p_rf + p_lr + p_linreg) / 4
              </span>
              <span>
                <span className="font-mono text-cyan-400">AUC-weighted:</span>
                &nbsp;ŷ = Σ (AUC_i / Σ AUC) × p_i
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Variant selector + weight tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Simple blend */}
        <div
          className={`card cursor-pointer transition-all ${activeTab === 'simple' ? 'border-purple-500/50' : ''}`}
          onClick={() => setActiveTab('simple')}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Scale size={15} className="text-purple-400" />
              <span className="text-sm font-semibold text-white">Simple Average</span>
            </div>
            <span
              className="badge text-xs"
              style={{ background: '#8b5cf622', color: '#a78bfa', border: '1px solid #8b5cf644' }}
            >
              Equal weights
            </span>
          </div>
          <WeightTable weights={simple.weights ?? []} color={VARIANT_COLORS.simple} />
          <MetricCards metrics={simple.metrics ?? {}} color={VARIANT_COLORS.simple} />
        </div>

        {/* AUC-weighted blend */}
        <div
          className={`card cursor-pointer transition-all ${activeTab === 'auc_weighted' ? 'border-cyan-500/50' : ''}`}
          onClick={() => setActiveTab('auc_weighted')}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp size={15} className="text-cyan-400" />
              <span className="text-sm font-semibold text-white">AUC-Weighted Average</span>
            </div>
            <span
              className="badge text-xs"
              style={{ background: '#06b6d422', color: '#22d3ee', border: '1px solid #06b6d444' }}
            >
              AUC-proportional
            </span>
          </div>
          <WeightTable weights={aucWtd.weights ?? []} color={VARIANT_COLORS.auc_weighted} />
          <MetricCards metrics={aucWtd.metrics ?? {}} color={VARIANT_COLORS.auc_weighted} />
        </div>
      </div>

      {/* F1 & AUC lift table */}
      <div className="card mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Layers size={15} className="text-purple-400" />
          <span className="text-sm font-semibold text-white">F1 &amp; AUC-ROC — All Models + Blends</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">Side-by-side comparison showing the lift from ensembling</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              {['Model', 'F1-Score', 'AUC-ROC'].map(h => (
                <th key={h} className="px-3 py-2 text-left text-slate-400 font-semibold text-xs">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {liftData.map((d, i) => {
              const isBlend = d.name.startsWith('Blend')
              return (
                <tr
                  key={i}
                  className={`border-b border-slate-800 ${isBlend ? 'bg-purple-500/5' : 'hover:bg-slate-800/30'}`}
                >
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: d.color }} />
                      <span className={`font-medium text-xs ${isBlend ? 'text-purple-300' : 'text-white'}`}>
                        {d.name}
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 font-mono text-xs text-slate-200">{d.f1.toFixed(4)}</td>
                  <td className="px-3 py-2.5 font-mono text-xs text-slate-200">{d.auc.toFixed(4)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* ROC curve including blends */}
      <div className="card mb-6">
        <p className="text-sm font-semibold text-white mb-1">ROC Curves — All Models + Blends</p>
        <p className="text-xs text-slate-400 mb-4">Blends shown with bold lines; base models dashed</p>
        <div className="flex flex-wrap gap-2 mb-3">
          {rocSeries.map(s => (
            <span
              key={s.key}
              className="badge text-[11px]"
              style={{ background: s.color + '22', color: s.color, border: `1px solid ${s.color}44` }}
            >
              {s.label}: {(s.key === 'simple' ? simple : s.key === 'auc_weighted' ? aucWtd : models[s.key])?.metrics?.roc_auc?.toFixed(4)}
            </span>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rocData} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="fpr" type="number" domain={[0, 1]}
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              label={{ value: 'FPR', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }}
              tickFormatter={v => v.toFixed(1)}
            />
            <YAxis
              domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }}
              label={{ value: 'TPR', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
              tickFormatter={v => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: '#94a3b8' }}
              formatter={(v, n) => [v?.toFixed(4), n]}
            />
            {rocSeries.map(s => {
              const isBlend = s.key === 'simple' || s.key === 'auc_weighted'
              return (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.label}
                  stroke={s.color}
                  strokeWidth={isBlend ? 2.5 : 1.5}
                  strokeDasharray={isBlend ? undefined : '5 3'}
                  dot={false}
                  connectNulls
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

    </section>
  )
}
