import React, { useState, useEffect } from 'react';
import { fetchHistory } from '../api';

export default function HistoryView() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchHistory()
      .then(res => {
        if (isMounted) setHistory(res);
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, []);

  if (loading) return <div className="card">Loading prediction history logs...</div>;
  if (error) return <div className="card error-banner">{error}</div>;

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Prediction History Log</h2>
      </div>

      {history.length === 0 ? (
        <div style={{ color: '#64748b', fontSize: '13px', padding: '16px 0' }}>
          No previous risk assessments recorded yet. Submit a test form from the Assessment tab.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="table-clinical">
            <thead>
              <tr>
                <th>ID</th>
                <th>Timestamp</th>
                <th>Condition</th>
                <th>Model Used</th>
                <th>Probability</th>
                <th>Risk Label</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {history.map(row => {
                const isExpanded = expandedId === row.id;
                const isHigh = row.risk_label === 'High Risk';
                return (
                  <React.Fragment key={row.id}>
                    <tr>
                      <td>#{row.id}</td>
                      <td style={{ fontSize: '12px', color: '#64748b' }}>{row.timestamp}</td>
                      <td style={{ textTransform: 'capitalize' }}>{row.condition}</td>
                      <td>{row.model_used}</td>
                      <td>{Math.round(row.probability * 100)}%</td>
                      <td>
                        <span className={`badge ${isHigh ? 'badge-high' : 'badge-low'}`}>
                          {row.risk_label}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="tab-btn"
                          style={{ padding: '4px 8px', fontSize: '11px' }}
                          onClick={() => setExpandedId(isExpanded ? null : row.id)}
                        >
                          {isExpanded ? 'Hide Inputs' : 'View Inputs'}
                        </button>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} style={{ background: '#f8fafc', padding: '12px' }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
                            Patient Clinical Inputs:
                          </div>
                          <pre style={{ fontSize: '11px', background: '#ffffff', padding: '8px', border: '1px solid #e2e8f0', borderRadius: '4px', overflowX: 'auto' }}>
                            {JSON.stringify(row.input_data, null, 2)}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
