import React, { useState, useEffect } from 'react';
import { fetchModelComparison } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ModelComparison() {
  const [condition, setCondition] = useState('heart');
  const [data, setData] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);
    fetchModelComparison(condition)
      .then(res => {
        if (isMounted) {
          setData(res);
          const firstModel = Object.keys(res.models)[0];
          setSelectedModel(firstModel);
        }
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [condition]);

  if (loading) return <div className="card">Loading comparative evaluation data...</div>;
  if (error) return <div className="card error-banner">{error}</div>;
  if (!data) return null;

  const models = data.models || {};
  const bestModelName = data.best_model_name;

  // Format ROC data for Recharts overlay chart
  const rocLines = Object.entries(models).map(([name, m]) => {
    const fpr = m.roc_curve?.fpr || [];
    const tpr = m.roc_curve?.tpr || [];
    const points = fpr.map((x, i) => ({ fpr: x, tpr: tpr[i] }));
    return { name, points, auc: m.roc_auc };
  });

  const selectedMetrics = selectedModel ? models[selectedModel] : null;
  const cm = selectedMetrics?.confusion_matrix || [[0, 0], [0, 0]];

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Side-by-Side Model Comparison</h2>
      </div>

      <div className="tab-group">
        {['heart', 'diabetes', 'liver'].map(c => (
          <button
            key={c}
            className={`tab-btn ${condition === c ? 'active' : ''}`}
            onClick={() => setCondition(c)}
          >
            {c.charAt(0).toUpperCase() + c.slice(1)} Dataset
          </button>
        ))}
      </div>

      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
          Cross-Validation & Test Metrics Table (Best Model: <span style={{ color: '#0f766e' }}>{bestModelName}</span>)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="table-clinical">
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Sensitivity (Recall)</th>
                <th>Specificity</th>
                <th>Precision</th>
                <th>F1 Score</th>
                <th>Cohen's Kappa</th>
                <th>ROC AUC</th>
                <th>SMOTE Recall</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(models).map(([name, m]) => {
                const isSelected = selectedModel === name;
                const isBest = name === bestModelName;
                return (
                  <tr
                    key={name}
                    onClick={() => setSelectedModel(name)}
                    style={{
                      cursor: 'pointer',
                      backgroundColor: isSelected ? '#f0fdf4' : isBest ? '#f8fafc' : 'transparent',
                      fontWeight: isBest ? 600 : 400
                    }}
                  >
                    <td>
                      {name} {isBest && <span style={{ fontSize: '10px', color: '#0f766e', marginLeft: '4px' }}>(Optimal)</span>}
                    </td>
                    <td style={{ color: m.sensitivity >= 0.85 ? '#166534' : 'inherit' }}>{m.sensitivity}</td>
                    <td>{m.specificity}</td>
                    <td>{m.precision}</td>
                    <td>{m.f1}</td>
                    <td>{m.cohen_kappa}</td>
                    <td>{m.roc_auc}</td>
                    <td>{m.smote_recall}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {selectedMetrics && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
              Confusion Matrix: {selectedModel}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', maxWidth: '300px' }}>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>True Negative (TN)</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#1e293b' }}>{cm[0][0]}</div>
              </div>
              <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '11px', color: '#991b1b' }}>False Positive (FP)</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#991b1b' }}>{cm[0][1]}</div>
              </div>
              <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '11px', color: '#991b1b' }}>False Negative (FN)</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#991b1b' }}>{cm[1][0]}</div>
              </div>
              <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '11px', color: '#166534' }}>True Positive (TP)</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#166534' }}>{cm[1][1]}</div>
              </div>
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
              ROC Curve Comparison Overview
            </h3>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              Selected model ({selectedModel}) achieved an ROC AUC of <strong>{selectedMetrics.roc_auc}</strong> and Sensitivity (Recall) of <strong>{selectedMetrics.sensitivity}</strong>.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
