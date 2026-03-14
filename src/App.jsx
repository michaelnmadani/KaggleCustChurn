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
    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginBottom: "28px" }}>
      {items.map((s) => (
        <div key={s.l} style={{
          flex: "1 1 120px", background: "#f9fafb", border: "1px solid #e5e7eb",
          borderRadius: "8px", padding: "16px", textAlign: "center",
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
        The dataset is based on the <strong>Kaggle Playground Series S6E3</strong> telco customer churn
        competition. It contains <strong>{stats.total_rows.toLocaleString()} customers</strong> and{" "}
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

      <Figure caption={`${steps.length} steps transform the raw data into ${data.meta.n_features} model-ready features`}>
        <div style={{ overflowX: "auto", border: "1px solid #e6e6e6", borderRadius: "6px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", ...sans }}>
            <thead>
              <tr>
                <th style={{ ...thStyle, width: "32px", textAlign: "center" }}>#</th>
                <th style={thStyle}>Step</th>
                <th style={thStyle}>Description</th>
                <th style={thStyle}>Code</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((row, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                  <td style={{ ...tdStyle, fontWeight: 700, color: "#999", textAlign: "center" }}>{i + 1}</td>
                  <td style={{ ...tdStyle, fontWeight: 600, color: "#222", whiteSpace: "nowrap" }}>{row.step}</td>
                  <td style={{ ...tdStyle, color: "#555", fontSize: "10px" }}>{row.description}</td>
                  <td style={tdStyle}>
                    <code style={{ ...mono, fontSize: "9px", background: "#f5f5f5", padding: "2px 6px", borderRadius: "3px", color: "#555" }}>
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

      <SubHeading>Data Loading & Cleaning Code</SubHeading>

      <CodeBlock>
        <span style={{ color: "#c586c0" }}>import</span>
        <span> pandas </span>
        <span style={{ color: "#c586c0" }}>as</span>
        <span> pd{"\n"}</span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span> numpy </span>
        <span style={{ color: "#c586c0" }}>as</span>
        <span> np{"\n"}</span>
        <span style={{ color: "#c586c0" }}>from</span>
        <span> sklearn.preprocessing </span>
        <span style={{ color: "#c586c0" }}>import</span>
        <span style={{ color: "#4ec9b0" }}> LabelEncoder</span>
        <span>{"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Load data</span>
        <span>{"\n"}</span>
        <span>df = pd.</span>
        <span style={{ color: "#dcdcaa" }}>read_csv</span>
        <span>(</span>
        <span style={{ color: "#ce9178" }}>"customer_churn.csv"</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 1: Drop customerID (unique identifier, no signal)</span>
        <span>{"\n"}</span>
        <span>df = df.</span>
        <span style={{ color: "#dcdcaa" }}>drop</span>
        <span>(columns=[</span>
        <span style={{ color: "#ce9178" }}>"customerID"</span>
        <span>]){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 2: TotalCharges stored as string — coerce to float</span>
        <span>{"\n"}</span>
        <span>df[</span>
        <span style={{ color: "#ce9178" }}>"TotalCharges"</span>
        <span>] = pd.</span>
        <span style={{ color: "#dcdcaa" }}>to_numeric</span>
        <span>(df[</span>
        <span style={{ color: "#ce9178" }}>"TotalCharges"</span>
        <span>], errors=</span>
        <span style={{ color: "#ce9178" }}>"coerce"</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 3: Impute {stats.missing_values.TotalCharges ?? 0} missing TotalCharges with median</span>
        <span>{"\n"}</span>
        <span>df[</span>
        <span style={{ color: "#ce9178" }}>"TotalCharges"</span>
        <span>] = df[</span>
        <span style={{ color: "#ce9178" }}>"TotalCharges"</span>
        <span>].</span>
        <span style={{ color: "#dcdcaa" }}>fillna</span>
        <span>(df[</span>
        <span style={{ color: "#ce9178" }}>"TotalCharges"</span>
        <span>].</span>
        <span style={{ color: "#dcdcaa" }}>median</span>
        <span>()){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 4: Encode binary target  Yes → 1, No → 0</span>
        <span>{"\n"}</span>
        <span>df[</span>
        <span style={{ color: "#ce9178" }}>"Churn"</span>
        <span>] = (df[</span>
        <span style={{ color: "#ce9178" }}>"Churn"</span>
        <span>] == </span>
        <span style={{ color: "#ce9178" }}>"Yes"</span>
        <span>).</span>
        <span style={{ color: "#dcdcaa" }}>astype</span>
        <span>(</span>
        <span style={{ color: "#4ec9b0" }}>int</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Step 5: Label-encode all remaining object columns</span>
        <span>{"\n"}</span>
        <span>le = </span>
        <span style={{ color: "#4ec9b0" }}>LabelEncoder</span>
        <span>(){"\n"}</span>
        <span style={{ color: "#c586c0" }}>for</span>
        <span> col </span>
        <span style={{ color: "#c586c0" }}>in</span>
        <span> df.</span>
        <span style={{ color: "#dcdcaa" }}>select_dtypes</span>
        <span>(include=</span>
        <span style={{ color: "#ce9178" }}>"object"</span>
        <span>).columns:{"\n"}</span>
        <span>{"    "}df[col] = le.</span>
        <span style={{ color: "#dcdcaa" }}>fit_transform</span>
        <span>(df[col].</span>
        <span style={{ color: "#dcdcaa" }}>astype</span>
        <span>(</span>
        <span style={{ color: "#4ec9b0" }}>str</span>
        <span>)){"\n\n"}</span>
        <span>X = df.</span>
        <span style={{ color: "#dcdcaa" }}>drop</span>
        <span>(columns=[</span>
        <span style={{ color: "#ce9178" }}>"Churn"</span>
        <span>]){"\n"}</span>
        <span>y = df[</span>
        <span style={{ color: "#ce9178" }}>"Churn"</span>
        <span>]</span>
      </CodeBlock>
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
        Four models are trained and compared: <strong>XGBoost</strong> (gradient-boosted trees),{" "}
        <strong>Random Forest</strong> (bagged decision trees), <strong>Logistic Regression</strong> (linear
        classifier), and <strong>Linear Regression</strong> repurposed as a probabilistic classifier. All are
        evaluated via <strong>{data.meta.cv_folds}-fold stratified cross-validation</strong> on the{" "}
        {data.meta.train_samples.toLocaleString()}-sample training set. Two blended ensembles average the
        out-of-fold probabilities from all four base models.
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
        { n: "4", l: "Base Models" },
      ]} />
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
    blend:  data.blend.simple_blend,
    wblend: data.blend.auc_weighted_blend,
  };
  const nameMap = {
    xgb:    "XGBoost",
    rf:     "Random Forest",
    lr:     "Logistic Regression",
    linreg: "Linear Regression",
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
        <span style={{ color: "#6a9955" }}># OOF probabilities from all four base models</span>
        <span>{"\n"}</span>
        <span>xgb_prob    = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(xgb_model,       X, y, cv=skf, method=</span>
        <span style={{ color: "#ce9178" }}>"predict_proba"</span>
        <span>)[:, </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>]{"\n"}</span>
        <span>lr_prob     = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(lr_pipeline,    X, y, cv=skf, method=</span>
        <span style={{ color: "#ce9178" }}>"predict_proba"</span>
        <span>)[:, </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>]{"\n"}</span>
        <span>rf_prob     = </span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(rf_model,       X, y, cv=skf, method=</span>
        <span style={{ color: "#ce9178" }}>"predict_proba"</span>
        <span>)[:, </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>]{"\n"}</span>
        <span>linreg_prob = np.</span>
        <span style={{ color: "#dcdcaa" }}>clip</span>
        <span>(</span>
        <span style={{ color: "#dcdcaa" }}>cross_val_predict</span>
        <span>(linreg_pipeline, X, y, cv=skf), </span>
        <span style={{ color: "#b5cea8" }}>0</span>
        <span>, </span>
        <span style={{ color: "#b5cea8" }}>1</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># Equal-weight soft-voting ensemble</span>
        <span>{"\n"}</span>
        <span>blend_prob = (xgb_prob + lr_prob + rf_prob + linreg_prob) / </span>
        <span style={{ color: "#b5cea8" }}>4.0</span>
        <span>{"\n"}</span>
        <span>blend_pred = (blend_prob {">="} </span>
        <span style={{ color: "#b5cea8" }}>0.5</span>
        <span>).</span>
        <span style={{ color: "#dcdcaa" }}>astype</span>
        <span>(</span>
        <span style={{ color: "#4ec9b0" }}>int</span>
        <span>){"\n\n"}</span>
        <span style={{ color: "#6a9955" }}># AUC-weighted ensemble — weight proportional to each model's AUC</span>
        <span>{"\n"}</span>
        <span>aucs    = [</span>
        <span style={{ color: "#dcdcaa" }}>roc_auc_score</span>
        <span>(y, p) </span>
        <span style={{ color: "#c586c0" }}>for</span>
        <span> p </span>
        <span style={{ color: "#c586c0" }}>in</span>
        <span> [xgb_prob, lr_prob, rf_prob, linreg_prob]]{"\n"}</span>
        <span>weights = np.</span>
        <span style={{ color: "#dcdcaa" }}>array</span>
        <span>(aucs) / </span>
        <span style={{ color: "#dcdcaa" }}>sum</span>
        <span>(aucs){"\n"}</span>
        <span>wblend  = </span>
        <span style={{ color: "#dcdcaa" }}>sum</span>
        <span>(w * p </span>
        <span style={{ color: "#c586c0" }}>for</span>
        <span> w, p </span>
        <span style={{ color: "#c586c0" }}>in</span>
        <span> </span>
        <span style={{ color: "#dcdcaa" }}>zip</span>
        <span>(weights, [xgb_prob, lr_prob, rf_prob, linreg_prob]))</span>
      </CodeBlock>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   SECTION 4 — Conclusion & Interpretation
═══════════════════════════════════════════════════════════ */

function ConclusionSection({ data }) {
  const [activeModel, setActiveModel] = useState("xgb");

  const modelMap = {
    xgb:    data.models.xgboost,
    rf:     data.models.random_forest,
    lr:     data.models.logistic_regression,
    linreg: data.models.linear_regression,
    blend:  data.blend.simple_blend,
    wblend: data.blend.auc_weighted_blend,
  };
  const nameMap = {
    xgb:    "XGBoost",
    rf:     "Random Forest",
    lr:     "Logistic Regression",
    linreg: "Linear Regression",
    blend:  "Blended (Equal Avg)",
    wblend: "Blended (AUC-Weighted)",
  };

  const m           = modelMap[activeModel];
  const modelName   = nameMap[activeModel];
  const topFeatures = (m.feature_importance || []).slice(0, 10);

  const xgbM  = data.models.xgboost.metrics;
  const rfM   = data.models.random_forest.metrics;
  const lrM   = data.models.logistic_regression.metrics;
  const linM  = data.models.linear_regression.metrics;
  const bM    = data.blend.simple_blend.metrics;
  const wbM   = data.blend.auc_weighted_blend.metrics;

  const aw   = data.blend.auc_weighted_blend;
  const wXgb = aw.weights.find((w) => w.model === "xgboost")?.weight ?? 0;
  const wLr  = aw.weights.find((w) => w.model === "logistic_regression")?.weight ?? 0;
  const wRf  = aw.weights.find((w) => w.model === "random_forest")?.weight ?? 0;
  const wLin = aw.weights.find((w) => w.model === "linear_regression")?.weight ?? 0;

  return (
    <section>
      <SectionHeading>4. Conclusion & Interpretation</SectionHeading>

      <P>
        All four models achieve comparable F1 on this heavily imbalanced dataset.
        XGBoost reaches <strong>{(xgbM.accuracy * 100).toFixed(1)}% accuracy</strong> (AUC {xgbM.roc_auc.toFixed(4)}),
        Random Forest <strong>{(rfM.accuracy * 100).toFixed(1)}%</strong> (AUC {rfM.roc_auc.toFixed(4)}),
        Logistic Regression <strong>{(lrM.accuracy * 100).toFixed(1)}%</strong> (AUC {lrM.roc_auc.toFixed(4)}),
        and Linear Regression <strong>{(linM.accuracy * 100).toFixed(1)}%</strong> (AUC {linM.roc_auc.toFixed(4)}).
        The <strong>equal-weight blend</strong> achieves <strong>{(bM.accuracy * 100).toFixed(1)}%</strong> (AUC {bM.roc_auc.toFixed(4)}),
        and the <strong>AUC-weighted blend</strong> achieves <strong>{(wbM.accuracy * 100).toFixed(1)}%</strong> (AUC {wbM.roc_auc.toFixed(4)},
        weights XGB:{wXgb.toFixed(3)} LR:{wLr.toFixed(3)} RF:{wRf.toFixed(3)} LinReg:{wLin.toFixed(3)}).
      </P>

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
             activeModel === "blend"  ? " (equal average of XGB + RF importances)" :
             activeModel === "wblend" ? " (AUC-weighted average of XGB + RF importances)" : ""}:
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
                ["Dataset",       `Kaggle S6E3 Churn — ${data.meta.train_samples.toLocaleString()} labelled customers`],
                ["Preprocessing", `${data.cleaning_steps.length} cleaning steps → ${data.meta.n_features} features`],
                ["Imbalance",     "scale_pos_weight (XGB) · class_weight='balanced' (RF, LR) · Youden's J (LinReg)"],
                ["Models",        "XGBoost · Random Forest · Logistic Regression · Linear Regression"],
                ["Ensembles",     "Equal-weight average + AUC-proportional weighted average"],
                ["Validation",    `${data.meta.cv_folds}-fold stratified cross-validation (OOF)`],
                ["XGBoost",       `Acc ${(xgbM.accuracy*100).toFixed(1)}%  F1 ${(xgbM.f1*100).toFixed(1)}%  AUC ${xgbM.roc_auc.toFixed(4)}`],
                ["Random Forest", `Acc ${(rfM.accuracy*100).toFixed(1)}%  F1 ${(rfM.f1*100).toFixed(1)}%  AUC ${rfM.roc_auc.toFixed(4)}`],
                ["Logistic Reg.", `Acc ${(lrM.accuracy*100).toFixed(1)}%  F1 ${(lrM.f1*100).toFixed(1)}%  AUC ${lrM.roc_auc.toFixed(4)}`],
                ["Linear Reg.",   `Acc ${(linM.accuracy*100).toFixed(1)}%  F1 ${(linM.f1*100).toFixed(1)}%  AUC ${linM.roc_auc.toFixed(4)}  thresh ${data.models.linear_regression.params.threshold}`],
                ["Equal Blend",   `Acc ${(bM.accuracy*100).toFixed(1)}%  F1 ${(bM.f1*100).toFixed(1)}%  AUC ${bM.roc_auc.toFixed(4)}`],
                ["AUC-Wtd Blend", `Acc ${(wbM.accuracy*100).toFixed(1)}%  F1 ${(wbM.f1*100).toFixed(1)}%  AUC ${wbM.roc_auc.toFixed(4)}`],
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

      {/* Bottom line */}
      <div style={{ background: "#111", borderRadius: "8px", padding: "24px 28px", marginTop: "32px", marginBottom: "8px" }}>
        <div style={{ ...sans, fontSize: "10px", fontWeight: 600, color: "#888", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Bottom Line
        </div>
        <div style={{ ...serif, fontSize: "15px", lineHeight: 1.7, color: "#e5e5e5" }}>
          XGBoost ({(xgbM.f1 * 100).toFixed(1)}% F1), Random Forest ({(rfM.f1 * 100).toFixed(1)}%),
          Logistic Regression ({(lrM.f1 * 100).toFixed(1)}%), and Linear Regression with Youden's J
          ({(linM.f1 * 100).toFixed(1)}%) all perform comparably on the imbalanced churn problem using{" "}
          {data.meta.n_features} engineered features and {data.meta.cv_folds}-fold CV. Contract type,
          tenure, and monthly charges dominate across all models. The AUC-weighted blend (
          {(wbM.f1 * 100).toFixed(1)}% F1, AUC {wbM.roc_auc.toFixed(4)}) is the top performer.
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
            Kaggle S6E3: Customer Churn Prediction
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
            <span><strong style={{ color: "#111" }}>{blendAcc}%</strong> Equal Blend</span>
            <span><strong style={{ color: "#111" }}>{wblendAcc}%</strong> AUC-Weighted</span>
            <span><strong style={{ color: "#111" }}>{data.meta.n_features}</strong> features</span>
            <span><strong style={{ color: "#111" }}>{data.meta.cv_folds}-fold</strong> CV</span>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: "720px", margin: "0 auto", padding: "0 24px 80px" }}>
        <CleaningSection data={data} />
        <Divider />
        <ModelSection data={data} />
        <Divider />
        <ResultsSection data={data} />
        <Divider />
        <ConclusionSection data={data} />
      </main>

      <footer style={{ borderTop: "1px solid #eee", textAlign: "center", padding: "32px 24px", ...sans, fontSize: "10px", color: "#bbb" }}>
        Kaggle S6E3: Customer Churn Prediction — XGBoost · Random Forest · Logistic Regression · Linear Regression · Blended Ensembles
      </footer>
    </div>
  );
}
