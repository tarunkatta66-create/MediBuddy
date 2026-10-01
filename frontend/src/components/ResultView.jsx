import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

export default function ResultView({ result, onBack }) {
  if (!result) return null;

  const { condition, probability, risk_label, model_used, top_shap_contributions, timestamp } = result;
  const isHighRisk = risk_label === 'High Risk';
  const probPercent = Math.round(probability * 100);

  const chartData = (top_shap_contributions || []).slice(0, 7).map(item => ({
    feature: item.feature,
    value: item.shap_value,
    rawVal: item.feature_value
  }));

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 className="card-title" style={{ textTransform: 'capitalize' }}>
            Risk Estimate Result: {condition} Condition
          </h2>
          <span style={{ fontSize: '12px', color: '#64748b' }}>Assessed at {timestamp}</span>
        </div>
        <button type="button" className="tab-btn" onClick={onBack}>New Assessment</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginBottom: '24px' }}>
        <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Estimated Risk Probability</div>
          <div style={{ fontSize: '32px', fontWeight: 700, color: isHighRisk ? '#991b1b' : '#166534', marginBottom: '8px' }}>
            {probPercent}%
          </div>
          <span className={`badge ${isHighRisk ? 'badge-high' : 'badge-low'}`}>
            {risk_label}
          </span>
          <div style={{ marginTop: '16px', fontSize: '12px', color: '#475569' }}>
            <strong>Selected Model:</strong> {model_used}
          </div>
        </div>

        <div>
          <h3 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#0f172a' }}>
            Feature Impact Breakdown (SHAP Values)
          </h3>
          <div style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer>
              <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(val, name, props) => [`${val} (Value: ${props.payload.rawVal})`, 'SHAP Value']}
                />
                <ReferenceLine x={0} stroke="#94a3b8" />
                <Bar dataKey="value">
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#dc2626' : '#2563eb'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="disclaimer-banner">
        <strong>Disclaimer:</strong> Educational prototype. Not a medical device. This tool is designed strictly for academic evaluation and clinical algorithm demonstration.
      </div>
    </div>
  );
}
