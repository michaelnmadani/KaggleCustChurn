import React, { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { LineChart as LineIcon } from 'lucide-react'

const MODEL_COLORS = {
  xgboost:             '#3b82f6',
  random_forest:       '#10b981',
  logistic_regression: '#f59e0b',
  linear_regression:   '#ec4899',
}

const MODEL_LABELS = {
  xgboost:             'XGBoost',
  random_forest:       'Random Forest',
  logistic_regression: 'Logistic Regression',
  linear_regression:   'Linear Regression',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">FPR: {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(4)}</p>
      ))}
    </div>
  )
}

const CalibTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">Mean Predicted: {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(4)}</p>
      ))}
    </div>
  )
}

export default function CalibrationSection({ data }) {
  const models = data.models ?? {}
  const [activeModels, setActiveModels] = useState(Object.keys(MODEL_COLORS))

  const toggleModel = (key) => {
    setActiveModels(prev =>
      prev.includes(key)
        ? prev.filter(k => k !== key)
        : [...prev, key]
    )
  }

  // Build unified ROC data (align by FPR index)
  const rocEntries = Object.entries(models)
    .filter(([k]) => activeModels.includes(k))
    .map(([k, v]) => ({
      key: k,
      fpr: v.metrics?.roc_curve?.fpr ?? [],
      tpr: v.metrics?.roc_curve?.tpr ?? [],
      auc: v.metrics?.roc_auc ?? 0,
    }))

  const maxLen = Math.max(...rocEntries.map(e => e.fpr.length), 0)
  const rocData = Array.from({ length: maxLen }, (_, i) => {
    const row = { fpr: rocEntries[0]?.fpr[i] ?? 0 }
    rocEntries.forEach(e => { row[e.key] = e.tpr[i] ?? null })
    return row
  })

  // Calibration data
  const calibEntries = Object.entries(models)
    .filter(([k]) => activeModels.includes(k))
    .map(([k, v]) => ({
      key: k,
      mp: v.metrics?.calibration_curve?.mean_predicted ?? [],
      fp: v.metrics?.calibration_curve?.fraction_positive ?? [],
    }))

  const maxCalibLen = Math.max(...calibEntries.map(e => e.mp.length), 0)
  const calibData = Array.from({ length: maxCalibLen }, (_, i) => {
    const row = { x: calibEntries[0]?.mp[i] ?? 0 }
    calibEntries.forEach(e => { row[e.key] = e.fp[i] ?? null })
    return row
  })

  // Perfect calibration reference line data
  const perfectCalib = [{ x: 0, perfect: 0 }, { x: 1, perfect: 1 }]

  return (
    <section id="calibration" className="mb-16 scroll-mt-6">
      <h2 className="section-title">Model Fit &amp; Calibration</h2>
      <p className="section-sub">
        ROC curves and reliability diagrams comparing all four models.
      </p>

      {/* Model selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {Object.entries(MODEL_LABELS).map(([key, label]) => {
          const active = activeModels.includes(key)
          return (
            <button
              key={key}
              onClick={() => toggleModel(key)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs
                          font-medium border transition-all ${
                active
                  ? 'border-transparent text-white'
                  : 'border-slate-600 text-slate-400 bg-transparent'
              }`}
              style={active ? { background: MODEL_COLORS[key] + '33', borderColor: MODEL_COLORS[key] + '88', color: MODEL_COLORS[key] } : {}}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: active ? MODEL_COLORS[key] : '#475569' }}
              />
              {label}
            </button>
          )
        })}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <div className="card">
          <div className="flex items-center gap-2 mb-1">
            <LineIcon size={15} className="text-blue-400" />
            <span className="text-sm font-semibold text-white">ROC Curve</span>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            True Positive Rate vs False Positive Rate
          </p>

          {/* AUC badges */}
          <div className="flex flex-wrap gap-2 mb-4">
            {Object.entries(models)
              .filter(([k]) => activeModels.includes(k))
              .map(([k, v]) => (
                <span
                  key={k}
                  className="badge text-[11px]"
                  style={{
                    background: MODEL_COLORS[k] + '22',
                    color: MODEL_COLORS[k],
                    border: `1px solid ${MODEL_COLORS[k]}44`,
                  }}
                >
                  {MODEL_LABELS[k]}: AUC {v.metrics?.roc_auc?.toFixed(4)}
                </span>
              ))
            }
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rocData} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="fpr"
                type="number" domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                label={{ value: 'False Positive Rate', position: 'insideBottom',
                         offset: -12, fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft',
                         fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                x={0} y={0} stroke="#475569" strokeDasharray="4 4"
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
              />
              {rocEntries.map(e => (
                <Line
                  key={e.key}
                  type="monotone"
                  dataKey={e.key}
                  name={MODEL_LABELS[e.key]}
                  stroke={MODEL_COLORS[e.key]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Calibration Curve */}
        <div className="card">
          <div className="flex items-center gap-2 mb-1">
            <LineIcon size={15} className="text-emerald-400" />
            <span className="text-sm font-semibold text-white">
              Calibration Curve (Reliability Diagram)
            </span>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Fraction of positives vs mean predicted probability — diagonal = perfect calibration
          </p>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={calibData.map((d, i) => ({
                ...d,
                perfect: perfectCalib[i <= 0 ? 0 : Math.min(i, 1)]?.perfect,
              }))}
              margin={{ top: 5, right: 10, left: 0, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="x"
                type="number" domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                label={{ value: 'Mean Predicted Probability', position: 'insideBottom',
                         offset: -12, fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                label={{ value: 'Fraction of Positives', angle: -90, position: 'insideLeft',
                         fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <Tooltip content={<CalibTooltip />} />
              {/* Perfect calibration */}
              <Line
                type="linear"
                data={[{ x: 0 }, { x: 1 }]}
                dataKey="x"
                name="Perfect"
                stroke="#475569"
                strokeDasharray="5 3"
                dot={false}
                strokeWidth={1.5}
              />
              {calibEntries.map(e => (
                <Line
                  key={e.key}
                  type="monotone"
                  dataKey={e.key}
                  name={MODEL_LABELS[e.key]}
                  stroke={MODEL_COLORS[e.key]}
                  strokeWidth={2}
                  dot={{ r: 3, fill: MODEL_COLORS[e.key] }}
                  connectNulls
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                formatter={v => <span style={{ color: '#cbd5e1' }}>{v}</span>}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  )
}
