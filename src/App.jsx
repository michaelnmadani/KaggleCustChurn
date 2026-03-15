import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
} from "recharts";
import resultsData from "./data/results.json";

/* ───── typography ───── */

const serif = { fontFamily: "'Source Serif 4', Georgia, 'Times New Roman', serif" };
const sans  = { fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif" };
const mono  = { fontFamily: "'Fira Code', 'Consolas', monospace" };

/* ───── helpers ───── */

function zipRoc(m) {
  const c = m.metrics.roc_curve;
  return c.fpr.map((f, i) => ({ fpr: f, tpr: c.tpr[i] }));
}

function zipCal(m) {
  const c = m.metrics.calibration_curve;
  return c.mean_predicted.map((p, i) => ({ predicted: p, actual: c.fraction_positive[i] }));
}

/* ───── reusable UI ───── */

function SectionHeading({ children }) {
  return (
    <h2 style={{ ...serif, fontSize: "28px", fontWeight: 700, color: "#242424", lineHeight: 1.3, marginBottom: "16px", marginTop: "56px" }}>
      {children}
    </h2>
  );
}

function SubHeading({ children }) {
  return (
    <h3 style={{ ...sans, fontSize: "20px", fontWeight: 700, color: "#242424", lineHeight: 1.4, marginBottom: "12px", marginTop: "32px" }}>
      {children}
    </h3>
  );
}

function P({ children, style: extra = {} }) {
  return (
    <p style={{ ...sans, fontSize: "13px", lineHeight: 1.8, color: "#555", marginBottom: "20px", ...extra }}>
      {children}
    </p>
  );
}

function Callout({ color = "#f0f7ff", border = "#b8d4f0", children }) {
  return (
    <div style={{
      background: color, borderLeft: `4px solid ${border}`, borderRadius: "4px",
      padding: "16px 20px", marginBottom: "24px",
      ...sans, fontSize: "12px", lineHeight: 1.7, color: "#333",
    }}>
      {children}
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <div style={{ background: "#1e1e1e", borderRadius: "6px", padding: "20px 24px", marginBottom: "28px", overflowX: "auto", border: "1px solid #333" }}>
      <pre style={{ ...mono, fontSize: "10.5px", lineHeight: 1.7, margin: 0, color: "#d4d4d4", whiteSpace: "pre-wrap", wordWrap: "break-word" }}>
        {children}
      </pre>
    </div>
  );
}

function Figure({ children, caption }) {
  return (
    <figure style={{ margin: "32px 0" }}>
      {children}
      {caption && (
        <figcaption style={{ ...sans, fontSize: "10px", color: "#999", textAlign: "center", marginTop: "10px", lineHeight: 1.5 }}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

function Divider() {
  return <div style={{ textAlign: "center", margin: "48px 0", color: "#ccc", letterSpacing: "8px", fontSize: "15px" }}>...</div>;
}

function StatRow({ items }) {
  return (
    <div style={{ display: "flex", gap: "12px", flexWrap: "nowrap", marginBottom: "28px" }}>
      {items.map((s) => (
        <div key={s.l} style={{
          flex: "1 1 0", minWidth: 0, background: "#f9fafb", border: "1px solid #e5e7eb",
          borderRadius: "8px", padding: "14px 10px", textAlign: "center",
        }}>
          <div style={{ ...sans, fontSize: "25px", fontWeight: 700, color: "#111" }}>{s.n}</div>
          <div style={{ ...sans, fontSize: "9px", color: "#888", marginTop: "4px" }}>{s.l}</div>
        </div>
      ))}
    </div>
  );
}

/* ───── table styles ───── */

const thStyle = {
  ...sans, fontSize: "9px", fontWeight: 600, textTransform: "uppercase",
  letterSpacing: "0.5px", padding: "12px 16px", textAlign: "left",
  borderBottom: "2px solid #222", color: "#333", background: "#fafafa",
};

const tdStyle = { ...sans, fontSize: "11px", padding: "10px 16px", borderBottom: "1px solid #eee", color: "#444" };

/* ───── model toggle ───── */

function ModelToggle({ active, onChange, showBlend = false }) {
  const models = [
    { id: "xgb",    label: "XGBoost" },
    { id: "rf",     label: "Random Forest" },
    { id: "lr",     label: "Logistic Reg." },
    { id: "linreg", label: "Linear Reg." },
    { id: "lgbm",   label: "LightGBM" },
    ...(showBlend ? [{ id: "blend", label: "Equal Blend" }, { id: "wblend", label: "AUC-Weighted" }] : []),
  ];
  return (
    <div style={{ display: "flex", marginBottom: "24px", flexWrap: "wrap" }}>
      {models.map(({ id, label }, idx) => {
        const isActive = active === id;
        const pos = idx === 0 ? "left" : idx === models.length - 1 ? "right" : "middle";
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            style={{
              ...sans, fontSize: "11px", fontWeight: isActive ? 700 : 500,
              padding: "8px 16px", border: "1px solid #ddd", cursor: "pointer",
              background: isActive ? "#111" : "#fff",
              color: isActive ? "#fff" : "#555",
              borderRadius: pos === "left" ? "6px 0 0 6px" : pos === "right" ? "0 6px 6px 0" : "0",
              borderLeft: pos !== "left" ? "none" : undefined,
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 0 — Introduction
═══════════════════════════════════════════════════════════ */

function IntroSection({ data }) {
  return (
    <section>
      <SectionHeading>0. Background & Data Source</SectionHeading>

      <SubHeading>What is Kaggle?</SubHeading>

      <P>
        <a href="https://www.kaggle.com" target="_blank" rel="noreferrer" style={{ color: "#111", fontWeight: 600 }}>Kaggle</a> is
        an online platform for data science competitions, public datasets, and shared notebooks.
        It is owned by Google and hosts thousands of competitions ranging from beginner-friendly
        exercises to high-stakes industry challenges with cash prizes. Competitors download a
        provided dataset, build a predictive model, and submit predictions to a public leaderboard
        that scores them against a hidden test set.
      </P>

      <P>
        Alongside full competitions, Kaggle runs a continuous series called{" "}
        <strong>Playground Series</strong> — shorter, lower-stakes exercises designed for
        learning and experimentation. Each episode (identified by season and episode number)
        ships a synthetic-but-realistic tabular dataset based on a real-world problem.
      </P>

      <SubHeading>This Dataset — Playground Series S6E3</SubHeading>

      <P>
        Season 6, Episode 3 of the Playground Series focuses on{" "}
        <strong>customer churn prediction</strong> for a telecommunications company.
        The dataset is synthetically generated from the well-known{" "}
        <strong>IBM Telco Customer Churn</strong> dataset (originally published on the
        IBM Watson Analytics community), which describes ~7,000 customers of a fictional
        US telco with demographic details, service subscriptions, billing information,
        and a binary churn label.
      </P>

      <P>
        The Kaggle version expands this to{" "}
        <strong>{data.meta.train_samples.toLocaleString()} labelled customers</strong> generated
        via a deep-learning synthesiser trained on the original IBM data, preserving realistic
        correlations between features while preventing direct copying. Each row represents one
        customer account.
      </P>

      <Callout color="#f9fafb" border="#d1d5db">
        <strong>Competition URL:</strong>{" "}
        <a href="https://www.kaggle.com/competitions/playground-series-s6e3" target="_blank" rel="noreferrer" style={{ color: "#111" }}>
          kaggle.com/competitions/playground-series-s6e3
        </a>
        <br />
        <strong>Original IBM dataset:</strong>{" "}
        <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank" rel="noreferrer" style={{ color: "#111" }}>
          kaggle.com/datasets/blastchar/telco-customer-churn
        </a>
      </Callout>

      <SubHeading>Feature Overview</SubHeading>

      <P>
        The raw dataset contains <strong>{data.dataset_stats.total_cols} columns</strong> across
        demographic, service, and billing categories:
      </P>

      <Figure>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Features</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Demographics",   "gender, SeniorCitizen, Partner, Dependents"],
                ["Account",        "tenure (months), Contract, PaperlessBilling, PaymentMethod"],
                ["Billing",        "MonthlyCharges, TotalCharges"],
                ["Phone services", "PhoneService, MultipleLines"],
                ["Internet",       "InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies"],
                ["Target",         "Churn — Yes / No"],
              ].map(([cat, feats], i) => (
                <tr key={cat} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: "#222", whiteSpace: "nowrap" }}>{cat}</td>
                  <td style={{ ...tdStyle, color: "#555" }}>{feats}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      <StatRow items={[
        { n: data.meta.train_samples.toLocaleString(), l: "Customers" },
        { n: data.dataset_stats.total_cols,            l: "Raw columns" },
        { n: data.meta.n_features,                     l: "Engineered features" },
        { n: `${(data.dataset_stats.churn_rate * 100).toFixed(1)}%`, l: "Churn rate" },
        { n: `${data.dataset_stats.class_distribution.Yes.toLocaleString()}`, l: "Churned" },
        { n: `${data.dataset_stats.class_distribution.No.toLocaleString()}`,  l: "Retained" },
      ]} />
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 1 — Data Cleaning & Feature Engineering
═══════════════════════════════════════════════════════════ */

function CleaningSection({ data }) {
  const raw   = data.raw_sample;
  const cols  = raw.length > 0 ? Object.keys(raw[0]) : [];
  const stats = data.dataset_stats;
  const steps = data.cleaning_steps;

  const missingData = Object.entries(stats.missing_values).map(([col, n]) => ({
    column: col, missing: n, total: stats.total_rows,
  }));

  const churnDist = [
    { label: "No Churn", count: stats.class_distribution.No,  fill: "#3b82f6" },
    { label: "Churned",  count: stats.class_distribution.Yes, fill: "#ef4444" },
  ];

  const ttStyle = { ...sans, borderRadius: 4, border: "1px solid #e5e7eb", fontSize: 9 };
  const tickY   = { ...sans, fontSize: 9, fill: "#999" };
  const tickX   = { ...sans, fontSize: 9, fill: "#555" };

  return (
    <section>
      <SectionHeading>1. Data Cleaning & Feature Engineering</SectionHeading>

      <P>
        The dataset is a telco customer churn dataset containing{" "}
        <strong>{stats.total_rows.toLocaleString()} customers</strong> and{" "}
        <strong>{stats.total_cols} raw columns</strong>. The target variable <em>Churn</em> is highly
        imbalanced at <strong>{(stats.churn_rate * 100).toFixed(1)}%</strong> positive — a key challenge
        for all models. All evaluation uses <strong>{data.meta.cv_folds}-fold stratified cross-validation</strong> to
        preserve the class ratio in every fold.
      </P>

      <SubHeading>Raw Dataset Sample</SubHeading>

      <Figure caption={`First 6 rows of the raw data — ${cols.length} columns, ${stats.total_rows.toLocaleString()} total rows`}>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>{cols.map((c) => <th key={c} style={{ ...thStyle, whiteSpace: "nowrap" }}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {raw.slice(0, 6).map((r, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  {cols.map((c) => (
                    <td key={c} style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                      {r[c] == null ? (
                        <span style={{ background: "#fee2e2", color: "#dc2626", padding: "2px 8px", borderRadius: "3px", fontSize: "9px", fontWeight: 600 }}>NULL</span>
                      ) : String(r[c]).slice(0, 28)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      <SubHeading>Class Distribution & Missing Values</SubHeading>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "32px" }}>
        <Figure caption="Target class distribution — severe imbalance at 10.4% churn">
          <div style={{ background: "#fff", border: "1px solid #e6e6e6", borderRadius: "6px", padding: "20px 8px 8px 0" }}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={churnDist} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="label" tick={tickX} />
                <YAxis tick={tickY} tickFormatter={(v) => v.toLocaleString()} />
                <Tooltip contentStyle={ttStyle} formatter={(v) => v.toLocaleString()} />
                <Bar dataKey="count" name="Customers" radius={[4, 4, 0, 0]}
                  fill="#3b82f6"
                  label={{ position: "top", ...sans, fontSize: 9, fill: "#555" }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Figure>

        <div>
          <p style={{ ...sans, fontSize: "12px", fontWeight: 700, color: "#111", marginBottom: "12px", marginTop: "20px" }}>Missing Values</p>
          {missingData.length === 0 ? (
            <p style={{ ...sans, fontSize: "11px", color: "#888" }}>No missing values in raw data.</p>
          ) : missingData.map((d) => (
            <div key={d.column} style={{ marginBottom: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                <span style={{ ...mono, fontSize: "10px", fontWeight: 600, color: "#555" }}>{d.column}</span>
                <span style={{ ...sans, fontSize: "9px", color: "#999" }}>
                  {d.missing} / {d.total} ({((d.missing / d.total) * 100).toFixed(1)}%)
                </span>
              </div>
              <div style={{ background: "#f0f0f0", borderRadius: "4px", height: "22px", overflow: "hidden" }}>
                <div style={{
                  width: `${(d.missing / d.total) * 100}%`, height: "100%",
                  background: "linear-gradient(90deg, #ef4444, #f87171)", borderRadius: "4px",
                  display: "flex", alignItems: "center", paddingLeft: "8px",
                }}>
                  <span style={{ ...sans, fontSize: "8px", color: "#fff", fontWeight: 700 }}>{d.missing}</span>
                </div>
              </div>
            </div>
          ))}
          <Callout color="#fef2f2" border="#ef4444">
            <strong>TotalCharges</strong> has {stats.missing_values.TotalCharges ?? 0} blank strings that
            fail numeric coercion. These are imputed with the column median after{" "}
            <code style={{ ...mono, fontSize: "10px" }}>pd.to_numeric(..., errors="coerce")</code>.
          </Callout>
        </div>
      </div>

      <SubHeading>Numeric Feature Summary</SubHeading>

      <Figure caption="Summary statistics for continuous features before cleaning">
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                {["Feature", "Mean", "Std", "Min", "Median", "Max", "Missing"].map((h) => (
                  <th key={h} style={{ ...thStyle, textAlign: h === "Feature" ? "left" : "center" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats.numeric_summaries.map((row, i) => (
                <tr key={row.feature} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>
                    <code style={{ ...mono, fontSize: "10px" }}>{row.feature}</code>
                  </td>
                  {["mean", "std", "min", "median", "max"].map((k) => (
                    <td key={k} style={{ ...tdStyle, textAlign: "center" }}>{row[k]?.toFixed(2)}</td>
                  ))}
                  <td style={{ ...tdStyle, textAlign: "center" }}>
                    {row.missing > 0 ? (
                      <span style={{ background: "#fee2e2", color: "#dc2626", padding: "2px 8px", borderRadius: "3px", fontSize: "9px", fontWeight: 600 }}>
                        {row.missing}
                      </span>
                    ) : <span style={{ color: "#10b981", fontWeight: 700 }}>0</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      <SubHeading>Cleaning Pipeline</SubHeading>

      <Figure caption={`${steps.length}-step unified pipeline — all ${data.meta.n_features} features used by every model`}>
        <div style={{ border: "1px solid #e6e6e6", borderRadius: "6px", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed", ...sans }}>
            <colgroup>
              <col style={{ width: "28px" }} />
              <col style={{ width: "18%" }} />
              <col style={{ width: "32%" }} />
              <col style={{ width: "50%" }} />
            </colgroup>
            <thead>
              <tr>
                <th style={{ ...thStyle, textAlign: "center" }}>#</th>
                <th style={thStyle}>Step</th>
                <th style={thStyle}>Description</th>
                <th style={thStyle}>Code</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((row, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 700, color: "#999", textAlign: "center" }}>{i + 1}</td>
                  <td style={{ ...tdStyle, fontWeight: 600, color: "#222", wordBreak: "break-word" }}>{row.step}</td>
                  <td style={{ ...tdStyle, color: "#555", fontSize: "10px", wordBreak: "break-word" }}>{row.description}</td>
                  <td style={{ ...tdStyle, wordBreak: "break-word" }}>
                    <code style={{ ...mono, fontSize: "9px", background: "#f5f5f5", padding: "2px 6px", borderRadius: "3px", color: "#555", whiteSpace: "pre-wrap" }}>
                      {row.code}
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      <Callout color="#fffbeb" border="#f59e0b">
        <strong>Class imbalance strategy:</strong> With only 10.4% churn, a naïve model predicts "no churn"
        everywhere and achieves 89.6% accuracy — while identifying exactly 0 churners. All classifiers
        use explicit imbalance handling:{" "}
        <code style={{ ...mono, fontSize: "10px" }}>scale_pos_weight</code> for XGBoost,{" "}
        <code style={{ ...mono, fontSize: "10px" }}>class_weight='balanced'</code> for Logistic Regression
        and Random Forest, and Youden's J threshold selection for Linear Regression.
      </Callout>

      <StatRow items={[
        { n: stats.total_rows.toLocaleString(), l: "Customers" },
        { n: `${(stats.churn_rate * 100).toFixed(1)}%`, l: "Churn Rate" },
        { n: String(data.meta.n_features), l: "Features" },
        { n: `${data.meta.cv_folds}-fold`, l: "CV Strategy" },
      ]} />

      <SubHeading>Unified Pipeline — clean_data() Code</SubHeading>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>import</span><span> pandas </span><span style={{ color: "#c586c0" }}>as</span><span> pd{"\n"}</span>
        <span style={{ color: "#c586c0" }}>import</span><span> numpy </span><span style={{ color: "#c586c0" }}>as</span><span> np{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span><span> sklearn.impute </span><span style={{ color: "#c586c0" }}>import</span><span style={{ color: "#4ec9b0" }}> SimpleImputer</span><span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span><span> sklearn.pipeline </span><span style={{ color: "#c586c0" }}>import</span><span style={{ color: "#4ec9b0" }}> Pipeline</span><span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span><span> sklearn.preprocessing </span><span style={{ color: "#c586c0" }}>import</span><span style={{ color: "#4ec9b0" }}> StandardScaler</span><span>{"\n\n"}</span>
        <span>df = pd.</span><span style={{ color: "#dcdcaa" }}>read_csv</span><span>(</span><span style={{ color: "#ce9178" }}>"customer_churn.csv"</span><span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 1: Drop customerID (unique identifier, no signal){"\n"}</span>
        <span>df = df.</span><span style={{ color: "#dcdcaa" }}>drop</span><span>(columns=[</span><span style={{ color: "#ce9178" }}>"customerID"</span><span>]){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 2: Encode binary target — Yes → 1, No → 0{"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"Churn"</span><span>] = df[</span><span style={{ color: "#ce9178" }}>"Churn"</span><span>].</span><span style={{ color: "#dcdcaa" }}>map</span><span>({"{"}</span><span style={{ color: "#ce9178" }}>"Yes"</span><span>: </span><span style={{ color: "#b5cea8" }}>1</span><span>, </span><span style={{ color: "#ce9178" }}>"No"</span><span>: </span><span style={{ color: "#b5cea8" }}>0</span><span>{"}"}).astype(int){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 3: TotalCharges → numeric (whitespace strings → NaN){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"TotalCharges"</span><span>] = pd.</span><span style={{ color: "#dcdcaa" }}>to_numeric</span><span>(df[</span><span style={{ color: "#ce9178" }}>"TotalCharges"</span><span>], errors=</span><span style={{ color: "#ce9178" }}>"coerce"</span><span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 4: Impute all missing numerics with column medians{"\n"}</span>
        <span>num_cols = [c </span><span style={{ color: "#c586c0" }}>for</span><span> c </span><span style={{ color: "#c586c0" }}>in</span><span> df.</span><span style={{ color: "#dcdcaa" }}>select_dtypes</span><span>(</span><span style={{ color: "#ce9178" }}>"number"</span><span>).columns </span><span style={{ color: "#c586c0" }}>if</span><span> c != </span><span style={{ color: "#ce9178" }}>"Churn"</span><span>]{"\n"}</span>
        <span>df[num_cols] = </span><span style={{ color: "#4ec9b0" }}>SimpleImputer</span><span>(strategy=</span><span style={{ color: "#ce9178" }}>"median"</span><span>).</span><span style={{ color: "#dcdcaa" }}>fit_transform</span><span>(df[num_cols]){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 5: Log-transform TotalCharges (log1p safe for zeros){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"TotalCharges"</span><span>] = np.</span><span style={{ color: "#dcdcaa" }}>log1p</span><span>(df[</span><span style={{ color: "#ce9178" }}>"TotalCharges"</span><span>]){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Steps 6–8: Feature flags (before OHE — strings still readable){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"TenureShort"</span><span>]   = (df[</span><span style={{ color: "#ce9178" }}>"tenure"</span><span>] &lt; </span><span style={{ color: "#b5cea8" }}>10</span><span>).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(int){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"TenureMid"</span><span>]     = ((df[</span><span style={{ color: "#ce9178" }}>"tenure"</span><span>] &gt;= </span><span style={{ color: "#b5cea8" }}>11</span><span>) &amp; (df[</span><span style={{ color: "#ce9178" }}>"tenure"</span><span>] &lt;= </span><span style={{ color: "#b5cea8" }}>20</span><span>)).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(int){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"TenureLong"</span><span>]    = (df[</span><span style={{ color: "#ce9178" }}>"tenure"</span><span>] &gt;= </span><span style={{ color: "#b5cea8" }}>21</span><span>).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(int){"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"HasFamilyTies"</span><span>] = ({"\n"}</span>
        <span>{"    "}(df[</span><span style={{ color: "#ce9178" }}>"Partner"</span><span>] == </span><span style={{ color: "#ce9178" }}>"Yes"</span><span>) &amp; (df[</span><span style={{ color: "#ce9178" }}>"Dependents"</span><span>] == </span><span style={{ color: "#ce9178" }}>"Yes"</span><span>){"\n"}</span>
        <span>).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(int){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 9: Log-transform tenure{"\n"}</span>
        <span>df[</span><span style={{ color: "#ce9178" }}>"log_tenure"</span><span>] = np.</span><span style={{ color: "#dcdcaa" }}>log1p</span><span>(df[</span><span style={{ color: "#ce9178" }}>"tenure"</span><span>]){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 10: One-hot encode all remaining categorical columns{"\n"}</span>
        <span>cat_cols = df.</span><span style={{ color: "#dcdcaa" }}>select_dtypes</span><span>(</span><span style={{ color: "#ce9178" }}>"object"</span><span>).columns.</span><span style={{ color: "#dcdcaa" }}>tolist</span><span>(){"\n"}</span>
        <span>df = pd.</span><span style={{ color: "#dcdcaa" }}>get_dummies</span><span>(df, columns=cat_cols, drop_first=</span><span style={{ color: "#569cd6" }}>True</span><span>){"\n"}</span>
        <span>df[df.</span><span style={{ color: "#dcdcaa" }}>select_dtypes</span><span>(</span><span style={{ color: "#ce9178" }}>"bool"</span><span>).columns] = df.</span><span style={{ color: "#dcdcaa" }}>select_dtypes</span><span>(</span><span style={{ color: "#ce9178" }}>"bool"</span><span>).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(int){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># StandardScaler applied inside Pipeline for LR &amp; LinReg only{"\n"}</span>
        <span>pipeline = </span><span style={{ color: "#4ec9b0" }}>Pipeline</span><span>([(</span><span style={{ color: "#ce9178" }}>"scaler"</span><span>, </span><span style={{ color: "#4ec9b0" }}>StandardScaler</span><span>()), (</span><span style={{ color: "#ce9178" }}>"model"</span><span>, estimator)]){"\n\n"}</span>
        <span>X = df.</span><span style={{ color: "#dcdcaa" }}>drop</span><span>(columns=[</span><span style={{ color: "#ce9178" }}>"Churn"</span><span>]).values  </span><span style={{ color: "#6a9955" }}># → shape ({data.meta.train_samples.toLocaleString()}, {data.meta.n_features}){"\n"}</span>
        <span>y = df[</span><span style={{ color: "#ce9178" }}>"Churn"</span><span>].values</span>
      </CodeBlock>

      <Callout color="#f0f7ff" border="#3b82f6">
        <strong>Why pairwise categorical combinations?</strong> The dataset has 15 categorical columns
        (gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod). Taking all pairs gives C(15, 2) = <strong>105 new integer-encoded
        features</strong> — e.g.{" "}
        <code style={{ ...mono, fontSize: "10px" }}>Contract_x_InternetService</code>{" "}
        encodes combinations like "Month-to-month + Fiber optic". All five models receive these
        features. Tree models (XGBoost, RF, LightGBM) exploit the joint split thresholds directly;
        LR and LinReg see them as integer covariates. "Month-to-month + fiber optic internet"
        has a far higher churn rate than either signal alone, and the pairwise feature exposes
        that joint effect directly to every model in the ensemble.
      </Callout>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 2 — Model Training
═══════════════════════════════════════════════════════════ */

function ModelSection({ data }) {
  const log       = data.models.xgboost.training_log;
  const xgbParams = data.models.xgboost.params;
  const linregThr = data.models.linear_regression.params.threshold;
  const churnRate = data.dataset_stats.churn_rate;

  return (
    <section>
      <SectionHeading>2. Model Training</SectionHeading>

      <P>
        Five models are trained and compared: <strong>XGBoost</strong> (gradient-boosted trees),{" "}
        <strong>Random Forest</strong> (bagged decision trees), <strong>Logistic Regression</strong> (linear
        classifier), <strong>Linear Regression</strong> repurposed as a probabilistic classifier, and{" "}
        <strong>LightGBM</strong> (histogram-based gradient boosting with a 10-step feature engineering
        pipeline). All are evaluated via <strong>{data.meta.cv_folds}-fold stratified cross-validation</strong>{" "}
        on the {data.meta.train_samples.toLocaleString()}-sample training set. Two blended ensembles average
        the out-of-fold probabilities from <strong>all five models</strong> — XGBoost, Random Forest,
        Logistic Regression, Linear Regression, and LightGBM.
      </P>

      <SubHeading>XGBoost</SubHeading>

      <P>
        XGBoost handles class imbalance via{" "}
        <code style={{ ...mono, fontSize: "11px" }}>scale_pos_weight</code>, which up-weights positive
        examples by the ratio of negatives to positives (~{Math.round((1 - churnRate) / churnRate)}×). A
        manual CV loop passes validation folds as{" "}
        <code style={{ ...mono, fontSize: "11px" }}>eval_set</code> for per-round loss tracking.
      </P>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> xgboost </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> XGBClassifier</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.model_selection </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> StratifiedKFold</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>cross_val_predict</span>
        <span>{"\n\n"}</span>
        <span>skf = </span>
        <span style={{ color: "#4ec9b0" }}>StratifiedKFold</span>
        <span>(n_splits=</span>
        <span style={{ color: "#b5cea8" }}>{data.meta.cv_folds}</span>
        <span>, shuffle=</span>
        <span style={{ color: "#569cd6" }}>True</span>
        <span>, random_state=</span>
        <span style={{ color: "#b5cea8" }}>42</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># scale_pos_weight = n_negative / n_positive to counter imbalance</span>
        <span>{"\n"}</span>
        <span>neg, pos = (y == </span>
        <span style={{ color: "#b5cea8" }}>0</span>
        <span>).</span>
        <span style={{ color: "#dcdcaa" }}>sum</span>
        <span>(), (y == </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>).</span>
        <span style={{ color: "#dcdcaa" }}>sum</span>
        <span>(){"\n\n"}</span>
        <span>xgb_model = </span>
        <span style={{ color: "#4ec9b0" }}>XGBClassifier</span>
        <span>({"\n"}</span>
        <span>{"    "}n_estimators=</span>
        <span style={{ color: "#b5cea8" }}>{xgbParams.n_estimators}</span>
        <span>, max_depth=</span>
        <span style={{ color: "#b5cea8" }}>{xgbParams.max_depth}</span>
        <span>, learning_rate=</span>
        <span style={{ color: "#b5cea8" }}>{xgbParams.learning_rate}</span>
        <span>,{"\n"}</span>
        <span>{"    "}subsample=</span>
        <span style={{ color: "#b5cea8" }}>{xgbParams.subsample}</span>
        <span>, colsample_bytree=</span>
        <span style={{ color: "#b5cea8" }}>{xgbParams.colsample_bytree}</span>
        <span>,{"\n"}</span>
        <span>{"    "}scale_pos_weight=neg/pos,{"\n"}</span>
        <span>{"    "}eval_metric=</span>
        <span style={{ color: "#ce9178" }}>"logloss"</span>
        <span>, random_state=</span>
        <span style={{ color: "#b5cea8" }}>42</span>
        <span>,{"\n"}</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Manual fold loop — passes eval_set for loss tracking</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>for</span>
        <span> train_idx, val_idx </span>
        <span style={{ color: "#c586c0" }}>in</span>
        <span> skf.</span>
        <span style={{ color: "#dcdcaa" }}>split</span>
        <span>(X, y):{"\n"}</span>
        <span>{"    "}m = </span>
        <span style={{ color: "#dcdcaa" }}>clone</span>
        <span>(xgb_model){"\n"}</span>
        <span>{"    "}m.</span>
        <span style={{ color: "#dcdcaa" }}>fit</span>
        <span>(X[train_idx], y[train_idx],{"\n"}</span>
        <span>{"        "}eval_set=[(X[val_idx], y[val_idx])], verbose=</span>
        <span style={{ color: "#569cd6" }}>False</span>
        <span>)</span>
      </CodeBlock>

      <SubHeading>Random Forest & Logistic Regression</SubHeading>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.ensemble </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> RandomForestClassifier</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.linear_model </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> LogisticRegression</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.pipeline </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> Pipeline</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.preprocessing </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> StandardScaler</span>
        <span>{"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># class_weight='balanced' re-weights each sample by inverse class frequency</span>
        <span>{"\n"}</span>
        <span>rf_model = </span>
        <span style={{ color: "#4ec9b0" }}>RandomForestClassifier</span>
        <span>({"\n"}</span>
        <span>{"    "}n_estimators=</span>
        <span style={{ color: "#b5cea8" }}>200</span>
        <span>, max_depth=</span>
        <span style={{ color: "#b5cea8" }}>10</span>
        <span>,{"\n"}</span>
        <span>{"    "}class_weight=</span>
        <span style={{ color: "#ce9178" }}>"balanced"</span>
        <span>, random_state=</span>
        <span style={{ color: "#b5cea8" }}>42</span>
        <span>,{"\n"}</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># LR inside a Pipeline so StandardScaler is applied within each CV fold</span>
        <span>{"\n"}</span>
        <span>lr_pipeline = </span>
        <span style={{ color: "#4ec9b0" }}>Pipeline</span>
        <span>([{"\n"}</span>
        <span>{"    "}(</span>
        <span style={{ color: "#ce9178" }}>"scaler"</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>StandardScaler</span>
        <span>()),{"\n"}</span>
        <span>{"    "}(</span>
        <span style={{ color: "#ce9178" }}>"model"</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>LogisticRegression</span>
        <span>({"\n"}</span>
        <span>{"        "}max_iter=</span>
        <span style={{ color: "#b5cea8" }}>1000</span>
        <span>, class_weight=</span>
        <span style={{ color: "#ce9178" }}>"balanced"</span>
        <span>,{"\n"}</span>
        <span>{"        "}solver=</span>
        <span style={{ color: "#ce9178" }}>"lbfgs"</span>
        <span>, random_state=</span>
        <span style={{ color: "#b5cea8" }}>42</span>
        <span>,{"\n"}</span>
        <span>{"    "})),{"\n"}</span>
        <span>]){"\n\n"}</span>
        <span>y_pred_rf = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(rf_model,    X, y, cv=skf){"\n"}</span>
        <span>y_pred_lr = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(lr_pipeline, X, y, cv=skf)</span>
      </CodeBlock>

      <SubHeading>Linear Regression + Youden's J Threshold</SubHeading>

      <P>
        OLS regression is repurposed as a probabilistic classifier: predictions are clipped to [0, 1] and
        the optimal decision threshold is selected automatically via <strong>Youden's J statistic</strong>{" "}
        (maximises TPR − FPR on the OOF ROC curve). With a 10.4% base rate, the standard 0.5 threshold
        would always predict "no churn" — Youden's J auto-selects <strong>{linregThr.toFixed(3)}</strong> instead,
        giving Linear Regression a competitive F1.
      </P>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.linear_model </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> LinearRegression</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.metrics </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> roc_curve</span>
        <span>{"\n\n"}</span>
        <span>linreg_pipeline = </span>
        <span style={{ color: "#4ec9b0" }}>Pipeline</span>
        <span>([{"\n"}</span>
        <span>{"    "}(</span>
        <span style={{ color: "#ce9178" }}>"scaler"</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>StandardScaler</span>
        <span>()),{"\n"}</span>
        <span>{"    "}(</span>
        <span style={{ color: "#ce9178" }}>"model"</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>LinearRegression</span>
        <span>(fit_intercept=</span>
        <span style={{ color: "#569cd6" }}>True</span>
        <span>)),{"\n"}</span>
        <span>]){"\n\n"}</span>
        <span>y_scores = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(linreg_pipeline, X, y, cv=skf, method=</span>
        <span style={{ color: "#ce9178" }}>"predict"</span>
        <span>){"\n"}</span>
        <span>y_scores = np.</span>
        <span style={{ color: "#dcdcaa" }}>clip</span>
        <span>(y_scores, </span>
        <span style={{ color: "#b5cea8" }}>0</span>
        <span>, </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Youden's J: find threshold that maximises TPR - FPR on OOF ROC</span>
        <span>{"\n"}</span>
        <span>fpr_arr, tpr_arr, thresholds = </span>
        <span style={{ color: "#dcdcaa" }}>roc_curve</span>
        <span>(y, y_scores){"\n"}</span>
        <span>best_idx = np.</span>
        <span style={{ color: "#dcdcaa" }}>argmax</span>
        <span>(tpr_arr - fpr_arr){"\n"}</span>
        <span style={{ color: "#6a9955" }}># Optimal threshold: {linregThr.toFixed(4)}  (vs 0.5 which always predicts 0)</span>
        <span>{"\n"}</span>
        <span>threshold = thresholds[best_idx]{"\n"}</span>
        <span>y_pred   = (y_scores {">="} threshold).</span>
        <span style={{ color: "#dcdcaa" }}>astype</span>
        <span>(</span>
        <span style={{ color: "#4ec9b0" }}>int</span>
        <span>)</span>
      </CodeBlock>

      <SubHeading>LightGBM + 10-Step Feature Engineering</SubHeading>

      <P>
        LightGBM uses a histogram-based split-finding algorithm that is significantly faster than
        exact-split GBDT and performs well on high-cardinality categorical features natively.
        Rather than using the standard cleaning pipeline, this model applies a dedicated{" "}
        <strong>10-step feature engineering pipeline</strong> producing <strong>{data.models.lightgbm?.n_features ?? 216} engineered features</strong>{" "}
        from the original {data.dataset_stats.total_cols} raw columns — including all C(15,2) = 105
        pairwise categorical combinations, an avg_charge_per_tenure ratio, and a log_avg_charge
        column. The decision threshold is
        chosen by grid-searching for the value that maximises the{" "}
        <strong>Matthews Correlation Coefficient (MCC)</strong> on OOF predictions — a better single
        metric for imbalanced binary classification than accuracy or F1 alone.
      </P>

      <Callout color="#f9fafb" border="#d1d5db">
        <strong>10-step pipeline ({data.models.lightgbm?.n_features ?? 216} engineered features):</strong>{" "}
        (1) Frequency encoding →
        (2) OOF target encoding →
        (3) RobustScaler →
        (4) KBins discretiser (10 bins) →
        (5) OrdinalEncoder →
        (6) <strong>All pairwise categorical combinations</strong> — C(15,2) = 105 pairs →
        (7) Interaction features (tenure×MonthlyCharges, SeniorCitizen×MonthlyCharges) →
        (8) Degree-2 polynomial features on numeric trio →
        (9) Ratio features: <code style={{ fontSize: "11px" }}>avg_charge_per_tenure</code> (TotalCharges ÷ tenure) +{" "}
        <code style={{ fontSize: "11px" }}>log_avg_charge</code> (log-compressed spend) →
        (10) Group aggregates (mean / median / std of numerics by Contract, InternetService, PaymentMethod)
      </Callout>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> lightgbm </span>
        <span style={{ color: "#c586c0" }}>as</span>
        <span> lgb{"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Params tuned for telco churn (AUC = 0.91367 on full dataset)</span>
        <span>{"\n"}</span>
        <span>params = &#123;{"\n"}</span>
        <span>{"    "}</span>
        <span style={{ color: "#ce9178" }}>"objective"</span>
        <span>: </span>
        <span style={{ color: "#ce9178" }}>"binary"</span>
        <span>, </span>
        <span style={{ color: "#ce9178" }}>"metric"</span>
        <span>: </span>
        <span style={{ color: "#ce9178" }}>"auc"</span>
        <span>,{"\n"}{"    "}</span>
        <span style={{ color: "#ce9178" }}>"n_estimators"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>3000</span>
        <span>, </span>
        <span style={{ color: "#ce9178" }}>"learning_rate"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>0.02</span>
        <span>,{"\n"}{"    "}</span>
        <span style={{ color: "#ce9178" }}>"num_leaves"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>20</span>
        <span>, </span>
        <span style={{ color: "#ce9178" }}>"max_depth"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>4</span>
        <span>,{"\n"}{"    "}</span>
        <span style={{ color: "#ce9178" }}>"subsample"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>0.7</span>
        <span>, </span>
        <span style={{ color: "#ce9178" }}>"colsample_bytree"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>0.7</span>
        <span>,{"\n"}{"    "}</span>
        <span style={{ color: "#ce9178" }}>"early_stopping_round"</span>
        <span>: </span>
        <span style={{ color: "#b5cea8" }}>300</span>
        <span>{"\n"}&#125;{"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Optimal threshold via MCC grid search (instead of fixed 0.5)</span>
        <span>{"\n"}</span>
        <span>best_mcc, best_t = -np.inf, </span>
        <span style={{ color: "#b5cea8" }}>0.5</span>
        <span>{"\n"}</span>
        <span style={{ color: "#c586c0" }}>for</span>
        <span> t </span>
        <span style={{ color: "#c586c0" }}>in</span>
        <span> np.</span>
        <span style={{ color: "#dcdcaa" }}>arange</span>
        <span>(</span>
        <span style={{ color: "#b5cea8" }}>0.05</span>
        <span>, </span>
        <span style={{ color: "#b5cea8" }}>0.96</span>
        <span>, </span>
        <span style={{ color: "#b5cea8" }}>0.01</span>
        <span>):{"\n"}</span>
        <span>{"    "}mcc = </span>
        <span style={{ color: "#dcdcaa" }}>matthews_corrcoef</span>
        <span>(y_oof, (oof_probs {">="} t).</span>
        <span style={{ color: "#dcdcaa" }}>astype</span>
        <span>(</span>
        <span style={{ color: "#4ec9b0" }}>int</span>
        <span>)){"\n"}</span>
        <span>{"    "}</span>
        <span style={{ color: "#c586c0" }}>if</span>
        <span> mcc {">"} best_mcc: best_mcc, best_t = mcc, t</span>
      </CodeBlock>

      <SubHeading>XGBoost Training Loss</SubHeading>

      <P>
        Log-loss measured on the full training set across {log.length} checkpoints (every 25 boosting
        rounds). The curve shows steady convergence — the model does not overfit within the given budget
        of {xgbParams.n_estimators} rounds.
      </P>

      <Figure caption={`XGBoost log-loss over ${xgbParams.n_estimators} boosting rounds (full training data)`}>
        <div style={{ background: "#fff", border: "1px solid #e6e6e6", borderRadius: "6px", padding: "20px 8px 8px 0" }}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={log} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="round" tick={{ ...sans, fontSize: 8, fill: "#999" }}
                label={{ value: "Boosting Round", position: "insideBottom", offset: -10, ...sans, fontSize: 8, fill: "#888" }} />
              <YAxis tick={{ ...sans, fontSize: 8, fill: "#999" }} domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ ...sans, borderRadius: 4, border: "1px solid #e5e7eb", fontSize: 9 }} />
              <Line type="monotone" dataKey="train_logloss" stroke="#333" name="Log Loss"
                dot={{ r: 3, fill: "#333" }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Figure>

      <StatRow items={[
        { n: data.meta.train_samples.toLocaleString(), l: "Training Samples" },
        { n: data.meta.n_features, l: "Input Features" },
        { n: `${data.meta.cv_folds}-fold`, l: "Cross-Validation" },
        { n: "5", l: "Base Models" },
      ]} />

      <SubHeading>Cross-Validation Fold Scores</SubHeading>
      <P>
        Per-fold accuracy, F1, and AUC across all {data.meta.cv_folds} stratified folds for each model.
        Consistent scores indicate stable models with low variance across splits.
      </P>

      {(() => {
        const models = [
          { key: "xgboost",            label: "XGBoost" },
          { key: "random_forest",      label: "Random Forest" },
          { key: "logistic_regression",label: "Logistic Reg." },
          { key: "linear_regression",  label: "Linear Reg." },
          { key: "lightgbm",           label: "LightGBM" },
        ];
        const foldCount = data.meta.cv_folds;
        const metrics = ["accuracy", "f1", "roc_auc"];
        const metricLabels = { accuracy: "Acc", f1: "F1", roc_auc: "AUC" };

        return (
          <Figure caption={`Each fold ≈ ${Math.round(data.meta.train_samples / foldCount)} customers. Metrics computed on held-out validation fold.`}>
            <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: "60px", textAlign: "center" }}>Fold</th>
                    {models.map(({ label }) =>
                      metrics.map((mk) => (
                        <th key={`${label}-${mk}`} style={{ ...thStyle, textAlign: "center", whiteSpace: "nowrap" }}>
                          {label}<br /><span style={{ fontWeight: 400, color: "#999" }}>{metricLabels[mk]}</span>
                        </th>
                      ))
                    )}
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: foldCount }, (_, fi) => (
                    <tr key={fi} style={{ background: fi % 2 === 0 ? "#fff" : "#fafafa" }}>
                      <td style={{ ...tdStyle, textAlign: "center", fontWeight: 700, color: "#555" }}>Fold {fi + 1}</td>
                      {models.map(({ key }) => {
                        const fs = (data.models[key].fold_scores || [])[fi] || {};
                        return metrics.map((mk) => (
                          <td key={`${key}-${mk}`} style={{ ...tdStyle, textAlign: "center", fontWeight: 600, color: "#111" }}>
                            {fs[mk] != null ? (mk === "roc_auc" ? fs[mk].toFixed(4) : `${(fs[mk] * 100).toFixed(1)}%`) : "—"}
                          </td>
                        ));
                      })}
                    </tr>
                  ))}
                  {/* Mean row */}
                  <tr style={{ background: "#f9fafb", borderTop: "2px solid #222" }}>
                    <td style={{ ...tdStyle, fontWeight: 700 }}>Mean</td>
                    {models.map(({ key }) => {
                      const foldScores = data.models[key].fold_scores || [];
                      return metrics.map((mk) => {
                        const vals = foldScores.map((f) => f[mk]).filter((v) => v != null);
                        const mean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
                        return (
                          <td key={`${key}-${mk}-mean`} style={{ ...tdStyle, textAlign: "center", fontWeight: 700, color: "#059669" }}>
                            {mean != null ? (mk === "roc_auc" ? mean.toFixed(4) : `${(mean * 100).toFixed(1)}%`) : "—"}
                          </td>
                        );
                      });
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          </Figure>
        );
      })()}
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 3 — Results & Validation
═══════════════════════════════════════════════════════════ */

function ResultsSection({ data }) {
  const [activeModel, setActiveModel] = useState("xgb");

  const modelMap = {
    xgb:    data.models.xgboost,
    rf:     data.models.random_forest,
    lr:     data.models.logistic_regression,
    linreg: data.models.linear_regression,
    lgbm:   data.models.lightgbm,
    blend:  data.blend.simple_blend,
    wblend: data.blend.auc_weighted_blend,
  };
  const nameMap = {
    xgb:    "XGBoost",
    rf:     "Random Forest",
    lr:     "Logistic Regression",
    linreg: "Linear Regression",
    lgbm:   "LightGBM",
    blend:  "Blended (Equal Avg)",
    wblend: "Blended (AUC-Weighted)",
  };

  const m         = modelMap[activeModel];
  const modelName = nameMap[activeModel];
  const { metrics } = m;
  const cm        = metrics.confusion_matrix;
  const total     = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
  const diagonal  = [{ predicted: 0, actual: 0 }, { predicted: 1, actual: 1 }];

  const metricsRows = [
    { metric: "Accuracy",  value: metrics.accuracy,  formula: "(TP + TN) / Total",    meaning: "Overall correct predictions" },
    { metric: "Precision", value: metrics.precision, formula: "TP / (TP + FP)",        meaning: "Of predicted churns, how many actually churned" },
    { metric: "Recall",    value: metrics.recall,    formula: "TP / (TP + FN)",        meaning: "Of actual churns, how many the model caught" },
    { metric: "F1 Score",  value: metrics.f1,        formula: "2 × P × R / (P + R)",  meaning: "Harmonic mean of precision & recall" },
  ];

  const allModels = [
    ["XGBoost",       data.models.xgboost.metrics],
    ["Random Forest", data.models.random_forest.metrics],
    ["Logistic Reg.", data.models.logistic_regression.metrics],
    ["Linear Reg.",   data.models.linear_regression.metrics],
    ...(data.models.lightgbm ? [["LightGBM", data.models.lightgbm.metrics]] : []),
    ["Equal Blend",   data.blend.simple_blend.metrics],
    ["AUC-Wtd Blend", data.blend.auc_weighted_blend.metrics],
  ];

  const rocData = zipRoc(m);
  const calData = zipCal(m);

  return (
    <section>
      <SectionHeading>3. Results & Validation</SectionHeading>

      <P>
        All models are evaluated on <strong>{data.meta.cv_folds}-fold stratified out-of-fold predictions</strong>.
        Use the toggle to inspect per-model metrics, confusion matrices, ROC curves, and calibration diagrams.
      </P>

      <SubHeading>Head-to-Head Comparison</SubHeading>

      <Figure>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                <th style={thStyle}>Metric</th>
                {allModels.map(([name]) => (
                  <th key={name} style={{ ...thStyle, textAlign: "center" }}>{name}</th>
                ))}
                <th style={{ ...thStyle, textAlign: "center" }}>Best</th>
              </tr>
            </thead>
            <tbody>
              {["accuracy", "precision", "recall", "f1", "roc_auc"].map((key, ri) => {
                const vals   = allModels.map(([, mt]) => mt[key]);
                const best   = Math.max(...vals);
                const labels = ["XGB", "RF", "LR", "LinReg", "Equal", "AUC-Wtd"];
                const winner = labels.filter((_, i) => vals[i] === best);
                const label  = winner.length === 1 ? winner[0] : "Tie";
                const display = { accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1 Score", roc_auc: "AUC" };
                return (
                  <tr key={key} style={{ background: ri % 2 === 0 ? "#fff" : "#fafafa" }}>
                    <td style={{ ...tdStyle, fontWeight: 600, color: "#222" }}>{display[key]}</td>
                    {vals.map((v, i) => (
                      <td key={i} style={{ ...tdStyle, textAlign: "center", fontWeight: 700, color: v === best ? "#059669" : "#111" }}>
                        {key === "roc_auc" ? v.toFixed(4) : `${(v * 100).toFixed(1)}%`}
                      </td>
                    ))}
                    <td style={{ ...tdStyle, textAlign: "center", fontSize: "9px", fontWeight: 600, color: "#888" }}>{label}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Figure>

      <SubHeading>{modelName} — Detailed Metrics</SubHeading>

      <ModelToggle active={activeModel} onChange={setActiveModel} showBlend />

      <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginBottom: "28px" }}>
        {metricsRows.map((mr) => (
          <div key={mr.metric} style={{ flex: "1 1 120px", background: "#111", borderRadius: "8px", padding: "20px 16px", textAlign: "center" }}>
            <div style={{ ...sans, fontSize: "29px", fontWeight: 700, color: "#fff" }}>{(mr.value * 100).toFixed(1)}%</div>
            <div style={{ ...sans, fontSize: "9px", color: "#999", marginTop: "4px", textTransform: "uppercase", letterSpacing: "0.5px" }}>{mr.metric}</div>
          </div>
        ))}
      </div>

      <Figure>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                <th style={thStyle}>Metric</th>
                <th style={{ ...thStyle, textAlign: "center" }}>Score</th>
                <th style={thStyle}>Formula</th>
                <th style={thStyle}>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {metricsRows.map((row, i) => (
                <tr key={row.metric} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: "#222" }}>{row.metric}</td>
                  <td style={{ ...tdStyle, textAlign: "center", fontWeight: 700, color: "#111" }}>{(row.value * 100).toFixed(1)}%</td>
                  <td style={tdStyle}>
                    <code style={{ ...mono, fontSize: "9px", background: "#f5f5f5", padding: "2px 6px", borderRadius: "3px" }}>{row.formula}</code>
                  </td>
                  <td style={{ ...tdStyle, color: "#666" }}>{row.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      <SubHeading>Confusion Matrix — {modelName}</SubHeading>

      <Figure caption={`${cm[0][0] + cm[1][1]} correct out of ${total} (${((cm[0][0] + cm[1][1]) / total * 100).toFixed(1)}%)`}>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: "8px", maxWidth: "380px", width: "100%" }}>
            <div />
            <div style={{ ...sans, fontSize: "8px", fontWeight: 700, color: "#888", textAlign: "center", textTransform: "uppercase", padding: "8px 0" }}>Pred: No Churn</div>
            <div style={{ ...sans, fontSize: "8px", fontWeight: 700, color: "#888", textAlign: "center", textTransform: "uppercase", padding: "8px 0" }}>Pred: Churned</div>

            <div style={{ ...sans, fontSize: "8px", fontWeight: 700, color: "#888", textTransform: "uppercase", display: "flex", alignItems: "center", justifyContent: "center", paddingRight: "8px", writingMode: "vertical-rl" }}>Actual: No Churn</div>
            <div style={{ background: "#059669", borderRadius: "8px", padding: "20px", textAlign: "center", color: "#fff" }}>
              <div style={{ ...sans, fontSize: "25px", fontWeight: 700 }}>{cm[0][0]}</div>
              <div style={{ fontSize: "8px", opacity: 0.8, marginTop: "4px" }}>True Negative</div>
            </div>
            <div style={{ background: "#ef4444", borderRadius: "8px", padding: "20px", textAlign: "center", color: "#fff" }}>
              <div style={{ ...sans, fontSize: "25px", fontWeight: 700 }}>{cm[0][1]}</div>
              <div style={{ fontSize: "8px", opacity: 0.8, marginTop: "4px" }}>False Positive</div>
            </div>

            <div style={{ ...sans, fontSize: "8px", fontWeight: 700, color: "#888", textTransform: "uppercase", display: "flex", alignItems: "center", justifyContent: "center", paddingRight: "8px", writingMode: "vertical-rl" }}>Actual: Churned</div>
            <div style={{ background: "#f97316", borderRadius: "8px", padding: "20px", textAlign: "center", color: "#fff" }}>
              <div style={{ ...sans, fontSize: "25px", fontWeight: 700 }}>{cm[1][0]}</div>
              <div style={{ fontSize: "8px", opacity: 0.8, marginTop: "4px" }}>False Negative</div>
            </div>
            <div style={{ background: "#059669", borderRadius: "8px", padding: "20px", textAlign: "center", color: "#fff" }}>
              <div style={{ ...sans, fontSize: "25px", fontWeight: 700 }}>{cm[1][1]}</div>
              <div style={{ fontSize: "8px", opacity: 0.8, marginTop: "4px" }}>True Positive</div>
            </div>
          </div>
        </div>
      </Figure>

      <SubHeading>Prediction Score Bands — {modelName}</SubHeading>

      {(() => {
        const bandKeyMap = {
          xgb:    "xgboost",
          rf:     "random_forest",
          lr:     "logistic_regression",
          linreg: "linear_regression",
          lgbm:   "lightgbm",
          blend:  "blend_simple",
          wblend: "blend_auc_weighted",
        };
        const bands = (data.score_bands ?? {})[bandKeyMap[activeModel]] ?? [];
        const totalCustomers = bands.reduce((s, b) => s + b.customers, 0);
        const baseRate = data.dataset_stats?.churn_rate ?? null;
        const maxChurn = Math.max(...bands.map(b => b.actual_churn_pct ?? 0), 1);
        return (
          <Figure>
            <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
                <thead>
                  <tr>
                    {["Predicted Band", "Customers", "% of Total", "Actual Churn %", "Lift vs Base"].map(h => (
                      <th key={h} style={thStyle}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bands.map((row, i) => {
                    const pctOfTotal = totalCustomers > 0
                      ? ((row.customers / totalCustomers) * 100).toFixed(1) + "%"
                      : "—";
                    const lift = baseRate && row.actual_churn_pct != null
                      ? (row.actual_churn_pct / 100 / baseRate).toFixed(2) + "×"
                      : "—";
                    const barPct = row.actual_churn_pct != null
                      ? Math.round((row.actual_churn_pct / maxChurn) * 100)
                      : 0;
                    return (
                      <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                        <td style={{ ...tdStyle, fontWeight: 600, color: "#222" }}>{row.band}</td>
                        <td style={{ ...tdStyle, textAlign: "right" }}>{row.customers.toLocaleString()}</td>
                        <td style={{ ...tdStyle, textAlign: "right", color: "#666" }}>{pctOfTotal}</td>
                        <td style={{ ...tdStyle }}>
                          {row.actual_churn_pct != null ? (
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <div style={{ flex: 1, background: "#e5e7eb", borderRadius: "4px", height: "6px", minWidth: "60px" }}>
                                <div style={{ width: `${barPct}%`, background: "#059669", height: "6px", borderRadius: "4px" }} />
                              </div>
                              <span style={{ fontWeight: 700, color: "#111", minWidth: "40px", textAlign: "right" }}>
                                {row.actual_churn_pct.toFixed(1)}%
                              </span>
                            </div>
                          ) : "—"}
                        </td>
                        <td style={{
                          ...tdStyle, textAlign: "right", fontWeight: 700,
                          color: lift !== "—" && parseFloat(lift) >= 2 ? "#059669"
                               : lift !== "—" && parseFloat(lift) >= 1 ? "#111"
                               : "#999",
                        }}>
                          {lift}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Figure>
        );
      })()}

      <SubHeading>{modelName} — ROC & Calibration</SubHeading>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "24px", marginBottom: "28px" }}>
        <Figure caption={`ROC Curve — AUC = ${metrics.roc_auc.toFixed(4)}`}>
          <div style={{ background: "#fff", border: "1px solid #e6e6e6", borderRadius: "6px", padding: "16px 8px 8px 0" }}>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={rocData} margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ ...sans, fontSize: 7, fill: "#999" }}
                  label={{ value: "False Positive Rate", position: "insideBottom", offset: -18, ...sans, fontSize: 8, fill: "#888" }} />
                <YAxis dataKey="tpr" type="number" domain={[0, 1]} tick={{ ...sans, fontSize: 7, fill: "#999" }}
                  label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", ...sans, fontSize: 8, fill: "#888" }} />
                <Tooltip contentStyle={{ ...sans, borderRadius: 4, fontSize: 11 }} />
                <Area type="monotone" dataKey="tpr" stroke="#333" fill="#e5e7eb" fillOpacity={0.5} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Figure>

        <Figure caption="Calibration — points near diagonal = well calibrated">
          <div style={{ background: "#fff", border: "1px solid #e6e6e6", borderRadius: "6px", padding: "16px 8px 8px 0" }}>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="predicted" type="number" domain={[0, 1]} tick={{ ...sans, fontSize: 7, fill: "#999" }}
                  label={{ value: "Predicted Probability", position: "insideBottom", offset: -18, ...sans, fontSize: 8, fill: "#888" }} />
                <YAxis dataKey="actual" type="number" domain={[0, 1]} tick={{ ...sans, fontSize: 7, fill: "#999" }}
                  label={{ value: "Actual Fraction Positive", angle: -90, position: "insideLeft", ...sans, fontSize: 8, fill: "#888" }} />
                <Tooltip contentStyle={{ ...sans, borderRadius: 4, fontSize: 11 }} />
                <Scatter data={calData} fill="#333" r={5} />
                <Scatter data={diagonal} fill="none" line={{ stroke: "#ccc", strokeDasharray: "5 5" }} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <SubHeading>Evaluation & Blending Code</SubHeading>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.metrics </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span> ({"\n"}</span>
        <span>{"    "}</span>
        <span style={{ color: "#4ec9b0" }}>accuracy_score</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>f1_score</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>roc_auc_score</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>confusion_matrix</span>
        <span>,{"\n"}</span>
        <span>{"    "}</span>
        <span style={{ color: "#4ec9b0" }}>precision_score</span>
        <span>, </span>
        <span style={{ color: "#4ec9b0" }}>recall_score</span>
        <span>,{"\n"}</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># OOF probabilities from all five models{"\n"}</span>
        <span>xgb_prob    = </span><span style={{ color: "#dcdcaa" }}>cross_val_predict</span><span>(xgb_model,       X, y, cv=skf, method=</span><span style={{ color: "#ce9178" }}>"predict_proba"</span><span>)[:, </span><span style={{ color: "#b5cea8" }}>1</span><span>]{"\n"}</span>
        <span>lr_prob     = </span><span style={{ color: "#dcdcaa" }}>cross_val_predict</span><span>(lr_pipeline,     X, y, cv=skf, method=</span><span style={{ color: "#ce9178" }}>"predict_proba"</span><span>)[:, </span><span style={{ color: "#b5cea8" }}>1</span><span>]{"\n"}</span>
        <span>rf_prob     = </span><span style={{ color: "#dcdcaa" }}>cross_val_predict</span><span>(rf_model,        X, y, cv=skf, method=</span><span style={{ color: "#ce9178" }}>"predict_proba"</span><span>)[:, </span><span style={{ color: "#b5cea8" }}>1</span><span>]{"\n"}</span>
        <span>linreg_prob = np.</span><span style={{ color: "#dcdcaa" }}>clip</span><span>(</span><span style={{ color: "#dcdcaa" }}>cross_val_predict</span><span>(linreg_pipeline, X, y, cv=skf), </span><span style={{ color: "#b5cea8" }}>0</span><span>, </span><span style={{ color: "#b5cea8" }}>1</span><span>){"\n"}</span>
        <span>lgbm_prob   = </span><span style={{ color: "#dcdcaa" }}>cross_val_predict</span><span>(lgbm_model,      X, y, cv=skf, method=</span><span style={{ color: "#ce9178" }}>"predict_proba"</span><span>)[:, </span><span style={{ color: "#b5cea8" }}>1</span><span>]{"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Equal-weight soft-voting ensemble (all 5 models){"\n"}</span>
        <span>all_probs  = [xgb_prob, lr_prob, rf_prob, linreg_prob, lgbm_prob]{"\n"}</span>
        <span>blend_prob = </span><span style={{ color: "#dcdcaa" }}>sum</span><span>(all_probs) / </span><span style={{ color: "#b5cea8" }}>5.0</span><span>{"\n"}</span>
        <span style={{ color: "#6a9955" }}># Threshold at 90th percentile → flag top 10% as predicted churn{"\n"}</span>
        <span>thresh     = np.</span><span style={{ color: "#dcdcaa" }}>percentile</span><span>(blend_prob, </span><span style={{ color: "#b5cea8" }}>90</span><span>)  </span>
        <span style={{ color: "#6a9955" }}># ≈ {data.blend.simple_blend.threshold}{"\n"}</span>
        <span>blend_pred = (blend_prob &gt;= thresh).</span><span style={{ color: "#dcdcaa" }}>astype</span><span>(</span><span style={{ color: "#4ec9b0" }}>int</span><span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># AUC-weighted ensemble — weight proportional to each model's OOF AUC{"\n"}</span>
        <span>aucs    = [</span><span style={{ color: "#dcdcaa" }}>roc_auc_score</span><span>(y, p) </span><span style={{ color: "#c586c0" }}>for</span><span> p </span><span style={{ color: "#c586c0" }}>in</span><span> all_probs]{"\n"}</span>
        <span>weights = np.</span><span style={{ color: "#dcdcaa" }}>array</span><span>(aucs) / </span><span style={{ color: "#dcdcaa" }}>sum</span><span>(aucs){"\n"}</span>
        <span>wblend  = </span><span style={{ color: "#dcdcaa" }}>sum</span><span>(w * p </span><span style={{ color: "#c586c0" }}>for</span><span> w, p </span><span style={{ color: "#c586c0" }}>in</span><span> </span><span style={{ color: "#dcdcaa" }}>zip</span><span>(weights, all_probs))</span>
      </CodeBlock>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 4 — Conclusion & Interpretation
═══════════════════════════════════════════════════════════ */

function ConclusionSection({ data }) {
  const [activeModel, setActiveModel] = useState("xgb");

  const aw    = data.blend.auc_weighted_blend;
  const wXgb  = aw.weights.find((w) => w.model === "xgboost")?.weight ?? 0;
  const wLr   = aw.weights.find((w) => w.model === "logistic_regression")?.weight ?? 0;
  const wRf   = aw.weights.find((w) => w.model === "random_forest")?.weight ?? 0;
  const wLin  = aw.weights.find((w) => w.model === "linear_regression")?.weight ?? 0;
  const wLgbm = aw.weights.find((w) => w.model === "lightgbm")?.weight ?? 0;

  // Blend feature importance: average XGB + RF + LGBM importances (the three tree models)
  const _mergeImportances = (fiArrays, weights) => {
    const map = {};
    fiArrays.forEach((fi, idx) => {
      const total = fi.reduce((s, f) => s + f.importance, 0) || 1;
      fi.forEach(({ feature, importance }) => {
        map[feature] = (map[feature] || 0) + (importance / total) * weights[idx];
      });
    });
    return Object.entries(map)
      .map(([feature, importance]) => ({ feature, importance }))
      .sort((a, b) => b.importance - a.importance);
  };
  const xgbFI  = data.models.xgboost.feature_importance || [];
  const rfFI   = data.models.random_forest.feature_importance || [];
  const lgbmFI = data.models.lightgbm?.feature_importance || [];
  const blendFI  = _mergeImportances([xgbFI, rfFI, lgbmFI], [1/3, 1/3, 1/3]);
  const wblendFI = _mergeImportances([xgbFI, rfFI, lgbmFI], [wXgb, wRf, wLgbm]);

  const modelMap = {
    xgb:    data.models.xgboost,
    rf:     data.models.random_forest,
    lr:     data.models.logistic_regression,
    linreg: data.models.linear_regression,
    lgbm:   data.models.lightgbm,
    blend:  { ...data.blend.simple_blend,       feature_importance: blendFI },
    wblend: { ...data.blend.auc_weighted_blend, feature_importance: wblendFI },
  };
  const nameMap = {
    xgb:    "XGBoost",
    rf:     "Random Forest",
    lr:     "Logistic Regression",
    linreg: "Linear Regression",
    lgbm:   "LightGBM",
    blend:  "Blended (Equal Avg)",
    wblend: "Blended (AUC-Weighted)",
  };

  const m           = modelMap[activeModel] ?? modelMap["xgb"];
  const modelName   = nameMap[activeModel]  ?? "XGBoost";
  const topFeatures = (m.feature_importance || []).slice(0, 10);

  const xgbM  = data.models.xgboost.metrics;
  const rfM   = data.models.random_forest.metrics;
  const lrM   = data.models.logistic_regression.metrics;
  const linM  = data.models.linear_regression.metrics;
  const lgbmM = data.models.lightgbm?.metrics;
  const bM    = data.blend.simple_blend.metrics;
  const wbM   = data.blend.auc_weighted_blend.metrics;

  return (
    <section>
      <SectionHeading>4. Summary & Interpretation</SectionHeading>

      <P>
        Five models are evaluated on a <strong>synthetic subset of {data.meta.train_samples.toLocaleString()} customers</strong>{" "}
        from a {(data.dataset_stats.churn_rate * 100).toFixed(1)}% churn-rate telco dataset ({" "}
        {data.meta.cv_folds}-fold stratified CV) using a <strong>unified {data.meta.n_features}-feature pipeline</strong>.
        All five models — XGBoost, Random Forest, Logistic Regression, Linear Regression, and LightGBM —
        are blended via soft-voting using equal-weight and AUC-proportional weighted averages of their
        out-of-fold predicted probabilities.
      </P>

      <Callout color="#fef9ec" border="#f59e0b">
        <strong>This pipeline runs on a synthetic 10,000-row subset</strong> of a larger telco churn
        dataset (~{(data.meta.train_samples * 44).toLocaleString()} rows total). AUC scores in the
        0.69–0.73 range are expected at this scale. On the full dataset, the LightGBM pipeline achieved{" "}
        <strong>0.91367 AUC</strong> — demonstrating that the feature engineering approach is sound and
        results should improve substantially with more data.
      </Callout>

      <Callout color="#faf5ff" border="#8b5cf6">
        <strong>Contract type is the top predictor across tree-based models.</strong> Month-to-month
        customers churn at far higher rates than one- or two-year contract holders. This is the strongest
        and most actionable business signal in the dataset.
      </Callout>

      <Callout color="#f0f7ff" border="#3b82f6">
        <strong>Tenure and MonthlyCharges are the strongest continuous signals.</strong> New customers
        with high monthly bills are the highest-risk cohort. Retention interventions during the first 6–12
        months have the highest expected ROI.
      </Callout>

      <Callout color="#ecfdf5" border="#10b981">
        <strong>Imbalance handling is critical.</strong> Without it, naïve classifiers achieve ~89.6%
        accuracy by predicting "no churn" for everyone — catching zero churners.
        F1 score and AUC, not accuracy, are the correct metrics for this problem.
      </Callout>

      <SubHeading>Feature Importance</SubHeading>

      <ModelToggle active={activeModel} onChange={setActiveModel} showBlend />

      {topFeatures.length === 0 ? (
        <P style={{ color: "#888" }}>
          Feature importance is not available for this model (Linear Regression via OLS does not produce
          Gini or gain-based scores). Switch to XGBoost, Random Forest, or a blend.
        </P>
      ) : (
        <>
          <P style={{ marginBottom: "12px" }}>
            Top features for <strong>{modelName}</strong>
            {activeModel === "xgb"    ? " (gain-based importance)" :
             activeModel === "rf"     ? " (Gini impurity importance)" :
             activeModel === "blend"  ? " (equal average of XGB + RF + LGBM importances)" :
             activeModel === "wblend" ? " (AUC-weighted average of XGB + RF + LGBM importances)" : ""}:
          </P>
          <Figure>
            <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, textAlign: "center", width: "50px" }}>#</th>
                    <th style={thStyle}>Feature</th>
                    <th style={thStyle}>Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {topFeatures.map((f, i) => (
                    <tr key={f.feature} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                      <td style={{ ...tdStyle, textAlign: "center", fontWeight: 700, color: i < 3 ? "#111" : "#999" }}>{i + 1}</td>
                      <td style={tdStyle}>
                        <code style={{ ...mono, fontSize: "10px", fontWeight: 500 }}>{f.feature}</code>
                      </td>
                      <td style={tdStyle}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <div style={{ width: "120px", background: "#f0f0f0", borderRadius: "3px", height: "14px", overflow: "hidden" }}>
                            <div style={{
                              width: `${(f.importance / topFeatures[0].importance) * 100}%`, height: "100%",
                              background: i < 3 ? "#111" : "#888", borderRadius: "3px",
                            }} />
                          </div>
                          <span style={{ ...sans, fontSize: "10px", fontWeight: 600, color: "#444" }}>
                            {(f.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Figure>
        </>
      )}

      <SubHeading>Pipeline Summary</SubHeading>

      <Figure>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                <th style={{ ...thStyle, width: "180px" }}>Component</th>
                <th style={thStyle}>Detail</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Dataset",       `Telco Customer Churn — synthetic ${data.meta.train_samples.toLocaleString()}-row subset (full dataset ≈ 440,000 rows)`],
                ["Pipeline",      `${data.cleaning_steps.length}-step unified pipeline → ${data.meta.n_features} features shared by all models`],
                ["Imbalance",     "scale_pos_weight (XGB) · class_weight='balanced' (RF, LR) · Youden's J (LinReg) · MCC grid (LGBM)"],
                ["Models",        "XGBoost · Random Forest · Logistic Regression · Linear Regression · LightGBM"],
                ["Ensembles",     "Equal-weight soft-vote + AUC-proportional weighted soft-vote (all 5 models)"],
                ["Validation",    `${data.meta.cv_folds}-fold stratified cross-validation (OOF predictions)`],
                ["XGBoost",       `Acc ${(xgbM.accuracy*100).toFixed(1)}%  F1 ${(xgbM.f1*100).toFixed(1)}%  AUC ${xgbM.roc_auc.toFixed(4)}`],
                ["Random Forest", `Acc ${(rfM.accuracy*100).toFixed(1)}%  F1 ${(rfM.f1*100).toFixed(1)}%  AUC ${rfM.roc_auc.toFixed(4)}`],
                ["Logistic Reg.", `Acc ${(lrM.accuracy*100).toFixed(1)}%  F1 ${(lrM.f1*100).toFixed(1)}%  AUC ${lrM.roc_auc.toFixed(4)}`],
                ["Linear Reg.",   `Acc ${(linM.accuracy*100).toFixed(1)}%  F1 ${(linM.f1*100).toFixed(1)}%  AUC ${linM.roc_auc.toFixed(4)}  thresh ${data.models.linear_regression.params.threshold}`],
                ...(lgbmM ? [["LightGBM",  `Acc ${(lgbmM.accuracy*100).toFixed(1)}%  F1 ${(lgbmM.f1*100).toFixed(1)}%  AUC ${lgbmM.roc_auc.toFixed(4)}  thresh ${data.models.lightgbm.params.optimal_threshold}`]] : []),
                ["Equal Blend",   `Acc ${(bM.accuracy*100).toFixed(1)}%  F1 ${(bM.f1*100).toFixed(1)}%  AUC ${bM.roc_auc.toFixed(4)}  thresh ${data.blend.simple_blend.threshold} (top 10%)`],
                ["AUC-Wtd Blend", `Acc ${(wbM.accuracy*100).toFixed(1)}%  F1 ${(wbM.f1*100).toFixed(1)}%  AUC ${wbM.roc_auc.toFixed(4)}  thresh ${data.blend.auc_weighted_blend.threshold} (top 10%)`],
              ].map(([label, detail], i) => (
                <tr key={label} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: "#222" }}>{label}</td>
                  <td style={{ ...tdStyle, color: "#555" }}>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>

      {/* Summary */}
      <div style={{ background: "#111", borderRadius: "8px", padding: "24px 28px", marginTop: "32px", marginBottom: "8px" }}>
        <div style={{ ...sans, fontSize: "10px", fontWeight: 600, color: "#888", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Summary
        </div>
        <div style={{ ...serif, fontSize: "15px", lineHeight: 1.7, color: "#e5e5e5" }}>
          These results reflect a <em>synthetic 10,000-row subset</em> of a larger telco churn dataset —
          not the full data. All five models share the same unified {data.meta.n_features}-feature pipeline.
          Individual model AUC scores range from{" "}
          {Math.min(xgbM.roc_auc, rfM.roc_auc, lrM.roc_auc, linM.roc_auc, lgbmM?.roc_auc ?? 1).toFixed(4)} to{" "}
          {Math.max(xgbM.roc_auc, rfM.roc_auc, lrM.roc_auc, linM.roc_auc, lgbmM?.roc_auc ?? 0).toFixed(4)},
          which is expected at this scale on a {(data.dataset_stats.churn_rate * 100).toFixed(1)}%-positive
          imbalanced task. The same feature engineering approach scored{" "}
          <strong style={{ color: "#fff" }}>0.91367 AUC</strong> on the full dataset — the gap is
          expected: pairwise interaction features require more data to generalise well.
        </div>
        <div style={{ ...serif, fontSize: "15px", lineHeight: 1.7, color: "#e5e5e5", marginTop: "14px" }}>
          The blended ensembles incorporate all five models. The equal-weight blend
          reaches AUC {bM.roc_auc.toFixed(4)} (F1 {(bM.f1 * 100).toFixed(1)}%). The AUC-weighted blend
          reaches AUC {wbM.roc_auc.toFixed(4)} (F1 {(wbM.f1 * 100).toFixed(1)}%) with weights
          XGB {wXgb.toFixed(3)} · LR {wLr.toFixed(3)} · RF {wRf.toFixed(3)} · LinReg {wLin.toFixed(3)} · LGBM {wLgbm.toFixed(3)}.
          Applying this unified pipeline to the full competition dataset is expected to meaningfully
          close the gap to the 0.91 AUC benchmark — more training examples give every model better
          signal, especially for the high-cardinality pairwise features.
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   Main App
═══════════════════════════════════════════════════════════ */

export default function App() {
  const data = resultsData;

  if (!data) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", ...sans }}>
        <div style={{ textAlign: "center", color: "#666" }}>
          <p style={{ fontWeight: 600, color: "#dc2626" }}>Error loading data</p>
          <p style={{ marginTop: "8px" }}>
            Run{" "}
            <code style={{ ...mono, background: "#f5f5f5", padding: "2px 6px", borderRadius: "3px" }}>
              python model_pipeline.py
            </code>{" "}
            first.
          </p>
        </div>
      </div>
    );
  }

  const xgbAcc    = (data.models.xgboost.metrics.accuracy * 100).toFixed(1);
  const rfAcc     = (data.models.random_forest.metrics.accuracy * 100).toFixed(1);
  const lrAcc     = (data.models.logistic_regression.metrics.accuracy * 100).toFixed(1);
  const linAcc    = (data.models.linear_regression.metrics.accuracy * 100).toFixed(1);
  const lgbmAuc   = data.models.lightgbm ? data.models.lightgbm.metrics.roc_auc.toFixed(4) : null;
  const blendAcc  = (data.blend.simple_blend.metrics.accuracy * 100).toFixed(1);
  const wblendAcc = (data.blend.auc_weighted_blend.metrics.accuracy * 100).toFixed(1);

  return (
    <div style={{ background: "#fff", minHeight: "100vh" }}>
      <header style={{ borderBottom: "1px solid #eee" }}>
        <div style={{ maxWidth: "720px", margin: "0 auto", padding: "60px 24px 40px" }}>
          <p style={{ ...sans, fontSize: "10px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "1.5px", color: "#999", marginBottom: "16px" }}>
            Machine Learning
          </p>
          <h1 style={{ ...serif, fontSize: "42px", fontWeight: 700, color: "#111", lineHeight: 1.2, marginBottom: "16px" }}>
            Customer Churn Prediction
          </h1>
          <p style={{ ...serif, fontSize: "17px", color: "#777", lineHeight: 1.5 }}>
            An end-to-end pipeline with imbalance handling, {data.meta.cv_folds}-fold cross-validation,
            and blended ensemble comparison across four classifiers
          </p>

          <div style={{ display: "flex", gap: "24px", marginTop: "28px", flexWrap: "wrap", ...sans, fontSize: "11px", color: "#555" }}>
            <span><strong style={{ color: "#111" }}>{xgbAcc}%</strong> XGBoost</span>
            <span><strong style={{ color: "#111" }}>{rfAcc}%</strong> Random Forest</span>
            <span><strong style={{ color: "#111" }}>{lrAcc}%</strong> Logistic Reg.</span>
            <span><strong style={{ color: "#111" }}>{linAcc}%</strong> Linear Reg.</span>
            {lgbmAuc && <span><strong style={{ color: "#111" }}>{lgbmAuc}</strong> LightGBM AUC</span>}
            <span><strong style={{ color: "#111" }}>{blendAcc}%</strong> Equal Blend</span>
            <span><strong style={{ color: "#111" }}>{wblendAcc}%</strong> AUC-Weighted</span>
            <span><strong style={{ color: "#111" }}>{data.meta.cv_folds}-fold</strong> CV</span>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: "720px", margin: "0 auto", padding: "0 24px 80px" }}>
        <IntroSection data={data} />
        <Divider />
        <CleaningSection data={data} />
        <Divider />
        <ModelSection data={data} />
        <Divider />
        <ResultsSection data={data} />
        <Divider />
        <ConclusionSection data={data} />
      </main>

      <footer style={{ borderTop: "1px solid #eee", textAlign: "center", padding: "32px 24px", ...sans, fontSize: "10px", color: "#bbb" }}>
        Customer Churn Prediction — XGBoost · Random Forest · Logistic Regression · Linear Regression · LightGBM · Blended Ensembles
      </footer>
    </div>
  );
}
