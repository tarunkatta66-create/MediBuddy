const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchConditionsSchema() {
  const res = await fetch(`${API_BASE}/conditions`);
  if (!res.ok) throw new Error('Failed to fetch conditions schema');
  return res.json();
}

export async function predictRisk(condition, formData) {
  const res = await fetch(`${API_BASE}/predict/${condition}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  });
  if (!res.ok) {
    const errorData = await res.json();
    throw new Error(errorData.detail || 'Prediction failed');
  }
  return res.json();
}

export async function fetchModelMetrics(condition) {
  const res = await fetch(`${API_BASE}/models/${condition}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics for ${condition}`);
  return res.json();
}

export async function fetchModelComparison(condition) {
  const res = await fetch(`${API_BASE}/models/${condition}/compare`);
  if (!res.ok) throw new Error(`Failed to fetch comparison for ${condition}`);
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch prediction history');
  return res.json();
}
