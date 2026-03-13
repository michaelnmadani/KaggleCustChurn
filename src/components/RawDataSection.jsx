import React, { useState, useMemo } from 'react'
import { Search, ChevronLeft, ChevronRight, TrendingUp, Users, AlertCircle } from 'lucide-react'

const PAGE_SIZE = 10

export default function RawDataSection({ data }) {
  const { raw_sample = [], dataset_stats = {} } = data
  const [page, setPage]       = useState(0)
  const [search, setSearch]   = useState('')
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')

  const columns = raw_sample.length > 0 ? Object.keys(raw_sample[0]) : []

  const filtered = useMemo(() => {
    let rows = raw_sample
    if (search.trim()) {
      const q = search.toLowerCase()
      rows = rows.filter(r =>
        Object.values(r).some(v => String(v ?? '').toLowerCase().includes(q))
      )
    }
    if (sortCol) {
      rows = [...rows].sort((a, b) => {
        const av = a[sortCol], bv = b[sortCol]
        if (av == null) return 1
        if (bv == null) return -1
        const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
        return sortDir === 'asc' ? cmp : -cmp
      })
    }
    return rows
  }, [raw_sample, search, sortCol, sortDir])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const paginated  = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('asc') }
    setPage(0)
  }

  const churnPct = ((dataset_stats.class_distribution?.Yes ?? 0) /
    (dataset_stats.total_rows ?? 1) * 100).toFixed(1)

  return (
    <section id="raw-data" className="mb-16 scroll-mt-6">
      <h2 className="section-title">Raw Data Overview</h2>
      <p className="section-sub">
        Telco Customer Churn dataset — {dataset_stats.total_rows?.toLocaleString()} records,&nbsp;
        {dataset_stats.total_cols} features
      </p>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="metric-card">
          <div className="metric-value">{dataset_stats.total_rows?.toLocaleString()}</div>
          <div className="metric-label">Total Rows</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{dataset_stats.total_cols}</div>
          <div className="metric-label">Features</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-amber-400">{churnPct}%</div>
          <div className="metric-label">Churn Rate</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-red-400">
            {Object.values(dataset_stats.missing_values ?? {}).reduce((s,v)=>s+v,0)}
          </div>
          <div className="metric-label">Missing Cells</div>
        </div>
      </div>

      {/* Class distribution & missing values */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="card">
          <div className="flex items-center gap-2 mb-3">
            <Users size={16} className="text-blue-400" />
            <span className="text-sm font-semibold text-white">Class Distribution</span>
          </div>
          <div className="space-y-2">
            {[
              { label: 'No Churn', key: 'No', color: 'bg-blue-500' },
              { label: 'Churned',  key: 'Yes', color: 'bg-red-500' },
            ].map(({ label, key, color }) => {
              const count = dataset_stats.class_distribution?.[key] ?? 0
              const pct   = ((count / (dataset_stats.total_rows ?? 1)) * 100).toFixed(1)
              return (
                <div key={key}>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>{label}</span>
                    <span>{count.toLocaleString()} ({pct}%)</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle size={16} className="text-amber-400" />
            <span className="text-sm font-semibold text-white">Missing Values</span>
          </div>
          {Object.keys(dataset_stats.missing_values ?? {}).length === 0 ? (
            <p className="text-sm text-slate-400">No missing values detected.</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(dataset_stats.missing_values ?? {}).map(([col, cnt]) => (
                <div key={col} className="flex justify-between text-sm">
                  <span className="text-slate-300 font-mono">{col}</span>
                  <span className="badge bg-amber-500/20 text-amber-400">{cnt} missing</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Interactive table */}
      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <div className="relative flex-1 max-w-sm">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search rows…"
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(0) }}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg pl-8 pr-3 py-2
                         text-sm text-slate-200 placeholder-slate-500 focus:outline-none
                         focus:border-blue-500 transition-colors"
            />
          </div>
          <span className="text-xs text-slate-500">{filtered.length} rows</span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-700/50">
          <table className="w-full text-xs min-w-max">
            <thead>
              <tr className="bg-slate-800/80">
                {columns.map(col => (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="px-3 py-2.5 text-left text-slate-300 font-semibold
                               cursor-pointer hover:text-white select-none whitespace-nowrap
                               border-b border-slate-700"
                  >
                    {col}
                    {sortCol === col && (
                      <span className="ml-1 text-blue-400">{sortDir === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginated.map((row, ri) => (
                <tr
                  key={ri}
                  className="border-b border-slate-800 hover:bg-slate-800/40 transition-colors"
                >
                  {columns.map(col => {
                    const val = row[col]
                    const isChurn = col === 'Churn'
                    return (
                      <td key={col} className="px-3 py-2 whitespace-nowrap">
                        {isChurn ? (
                          <span className={`badge ${
                            val === 1 || val === 'Yes'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-green-500/20 text-green-400'
                          }`}>
                            {val === 1 || val === 'Yes' ? 'Yes' : 'No'}
                          </span>
                        ) : val == null ? (
                          <span className="text-amber-500/70 italic">null</span>
                        ) : (
                          <span className="text-slate-300">{String(val)}</span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-slate-500">
            Page {page + 1} of {totalPages} · Showing {paginated.length} of {filtered.length}
          </span>
          <div className="flex gap-1">
            <button
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
              className="p-1.5 rounded bg-slate-800 text-slate-400 hover:text-white
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(p => p + 1)}
              className="p-1.5 rounded bg-slate-800 text-slate-400 hover:text-white
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Numeric summaries */}
      {dataset_stats.numeric_summaries?.length > 0 && (
        <div className="card mt-4">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp size={16} className="text-blue-400" />
            <span className="text-sm font-semibold text-white">Numeric Feature Summaries</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-700">
                  {['Feature','Mean','Std Dev','Min','Median','Max','Missing'].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-slate-400 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset_stats.numeric_summaries.map(s => (
                  <tr key={s.feature} className="border-b border-slate-800 hover:bg-slate-800/30">
                    <td className="px-3 py-2 font-mono text-slate-300">{s.feature}</td>
                    <td className="px-3 py-2 text-slate-400">{s.mean}</td>
                    <td className="px-3 py-2 text-slate-400">{s.std}</td>
                    <td className="px-3 py-2 text-slate-400">{s.min}</td>
                    <td className="px-3 py-2 text-slate-400">{s.median}</td>
                    <td className="px-3 py-2 text-slate-400">{s.max}</td>
                    <td className="px-3 py-2">
                      {s.missing > 0
                        ? <span className="badge bg-amber-500/20 text-amber-400">{s.missing}</span>
                        : <span className="badge bg-green-500/20 text-green-400">0</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
