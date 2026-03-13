import React, { useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, ResponsiveContainer, Legend, Cell,
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

const MetricTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-200 font-semibold mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.fill || p.stroke }}>
          {p.name}: {typeof p.value === 'number' ? (p.value > 1 ? p.value.toFixed(1) + '%' : p.value.toFixed(4)) : p.value}
        </p>
      ))}
    </div>
  )
}

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

  // Radar: blend vs each base model — accuracy, precision, recall, f1, auc
  const METRICS_RADAR = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
  const radarData = METRICS_RADAR.map(m => {
    const row = { metric: m.replace('roc_auc', 'AUC').toUpperCase() }
    Object.entries(models).forEach(([k, v]) => {
      row[MODEL_DISPLAY[k]] = parseFloat(((v.metrics?.[m] ?? 0) * 100).toFixed(1))
    })
    row['Blend (Simple)']   = parseFloat(((simple.metrics?.[m] ?? 0) * 100).toFixed(1))
    row['Blend (AUC-Wtd)']  = parseFloat(((aucWtd.metrics?.[m] ?? 0) * 100).toFixed(1))
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

      {/* F1 & AUC lift chart */}
      <div className="card mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Layers size={15} className="text-purple-400" />
          <span className="text-sm font-semibold text-white">F1 &amp; AUC-ROC — All Models + Blends</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">Side-by-side comparison showing the lift from ensembling</p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={liftData} margin={{ top: 5, right: 10, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              angle={-30} textAnchor="end"
              interval={0}
            />
            <YAxis
              domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }}
              tickFormatter={v => v.toFixed(1)}
            />
            <Tooltip content={<MetricTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
              formatter={v => <span style={{ color: '#cbd5e1' }}>{v}</span>}
            />
            <Bar dataKey="f1" name="F1-Score" radius={[3, 3, 0, 0]}>
              {liftData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.9} />)}
            </Bar>
            <Bar dataKey="auc" name="AUC-ROC" radius={[3, 3, 0, 0]}>
              {liftData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.45} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
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

      {/* Radar */}
      <div className="card">
        <p className="text-sm font-semibold text-white mb-1">Multi-Metric Radar — Blends vs Base Models</p>
        <p className="text-xs text-slate-400 mb-4">Values as percentages (0–100)</p>
        <ResponsiveContainer width="100%" height={360}>
          <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
            <PolarGrid stroke="#1e293b" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 9 }} tickCount={4} />
            {Object.entries(models).map(([k], i) => (
              <Radar
                key={k}
                name={MODEL_DISPLAY[k]}
                dataKey={MODEL_DISPLAY[k]}
                stroke={BASE_COLORS[i]}
                fill={BASE_COLORS[i]}
                fillOpacity={0.08}
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            ))}
            <Radar name="Blend (Simple)"   dataKey="Blend (Simple)"   stroke={VARIANT_COLORS.simple}       fill={VARIANT_COLORS.simple}       fillOpacity={0.18} strokeWidth={2.5} />
            <Radar name="Blend (AUC-Wtd)" dataKey="Blend (AUC-Wtd)" stroke={VARIANT_COLORS.auc_weighted} fill={VARIANT_COLORS.auc_weighted} fillOpacity={0.18} strokeWidth={2.5} />
            <Legend wrapperStyle={{ fontSize: 11 }} formatter={v => <span style={{ color: '#cbd5e1' }}>{v}</span>} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
