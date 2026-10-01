import React, { useState } from 'react';
import { predictRisk } from '../api';

const DEFAULT_VALS = {
  heart: {
    age: 55, sex: 1, cp: 2, trestbps: 130, chol: 240, fbs: 0,
    restecg: 1, thalach: 150, exang: 0, oldpeak: 1.2, slope: 1, ca: 0, thal: 2
  },
  diabetes: {
    Pregnancies: 2, Glucose: 120, BloodPressure: 70, SkinThickness: 20,
    Insulin: 80, BMI: 25.5, DiabetesPedigreeFunction: 0.45, Age: 40
  },
  liver: {
    Age: 45, Gender: 1, Total_Bilirubin: 1.2, Direct_Bilirubin: 0.4,
    Alkaline_Phosphotase: 180, Alamine_Aminotransferase: 30,
    Aspartate_Aminotransferase: 35, Total_Protiens: 6.8, Albumin: 3.4,
    Albumin_and_Globulin_Ratio: 1.0
  }
};

const FIELD_CONFIGS = {
  heart: [
    { name: 'age', label: 'Age', type: 'number', min: 1, max: 120, step: 1, unit: 'years', hint: 'Patient age in years' },
    { name: 'sex', label: 'Sex', type: 'select', options: [{ label: 'Male (1)', value: 1 }, { label: 'Female (0)', value: 0 }], hint: 'Biological sex' },
    { name: 'cp', label: 'Chest Pain Type', type: 'select', options: [{ label: 'Typical Angina (0)', value: 0 }, { label: 'Atypical Angina (1)', value: 1 }, { label: 'Non-anginal Pain (2)', value: 2 }, { label: 'Asymptomatic (3)', value: 3 }], hint: 'Chest pain classification' },
    { name: 'trestbps', label: 'Resting Blood Pressure', type: 'number', min: 60, max: 260, step: 1, unit: 'mmHg', hint: 'Resting BP on admission' },
    { name: 'chol', label: 'Serum Cholesterol', type: 'number', min: 80, max: 600, step: 1, unit: 'mg/dl', hint: 'Serum cholesterol level' },
    { name: 'fbs', label: 'Fasting Blood Sugar > 120 mg/dl', type: 'select', options: [{ label: 'True (>120 mg/dl)', value: 1 }, { label: 'False (<=120 mg/dl)', value: 0 }], hint: 'Fasting blood sugar category' },
    { name: 'restecg', label: 'Resting ECG Results', type: 'select', options: [{ label: 'Normal (0)', value: 0 }, { label: 'ST-T Wave Abnormality (1)', value: 1 }, { label: 'Left Ventricular Hypertrophy (2)', value: 2 }], hint: 'ECG baseline reading' },
    { name: 'thalach', label: 'Max Heart Rate', type: 'number', min: 60, max: 220, step: 1, unit: 'bpm', hint: 'Maximum heart rate achieved' },
    { name: 'exang', label: 'Exercise Angina', type: 'select', options: [{ label: 'Yes (1)', value: 1 }, { label: 'No (0)', value: 0 }], hint: 'Angina induced by exercise' },
    { name: 'oldpeak', label: 'ST Depression (Oldpeak)', type: 'number', min: 0, max: 10, step: 0.1, unit: 'mm', hint: 'ST depression induced by exercise' },
    { name: 'slope', label: 'ST Slope', type: 'select', options: [{ label: 'Upsloping (0)', value: 0 }, { label: 'Flat (1)', value: 1 }, { label: 'Downsloping (2)', value: 2 }], hint: 'Slope of peak exercise ST segment' },
    { name: 'ca', label: 'Major Vessels (ca)', type: 'number', min: 0, max: 4, step: 1, hint: 'Vessels colored by fluoroscopy (0-4)' },
    { name: 'thal', label: 'Thalassemia (thal)', type: 'select', options: [{ label: 'Normal (0)', value: 0 }, { label: 'Fixed Defect (1)', value: 1 }, { label: 'Reversible Defect (2)', value: 2 }, { label: 'Unknown (3)', value: 3 }], hint: 'Blood disorder reading' }
  ],
  diabetes: [
    { name: 'Age', label: 'Age', type: 'number', min: 1, max: 120, step: 1, unit: 'years', hint: 'Patient age' },
    { name: 'Pregnancies', label: 'Pregnancies', type: 'number', min: 0, max: 20, step: 1, hint: 'Number of times pregnant' },
    { name: 'Glucose', label: 'Glucose', type: 'number', min: 40, max: 400, step: 1, unit: 'mg/dl', hint: '2-hour plasma glucose concentration' },
    { name: 'BloodPressure', label: 'Diastolic Blood Pressure', type: 'number', min: 30, max: 200, step: 1, unit: 'mmHg', hint: 'Diastolic blood pressure' },
    { name: 'SkinThickness', label: 'Triceps Skin Thickness', type: 'number', min: 5, max: 100, step: 1, unit: 'mm', hint: 'Triceps skin fold thickness' },
    { name: 'Insulin', label: '2-Hour Serum Insulin', type: 'number', min: 5, max: 900, step: 1, unit: 'mu U/ml', hint: 'Serum insulin level' },
    { name: 'BMI', label: 'Body Mass Index (BMI)', type: 'number', min: 10, max: 70, step: 0.1, unit: 'kg/m²', hint: 'Body mass index' },
    { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', type: 'number', min: 0.05, max: 3, step: 0.01, hint: 'Genetic score function' }
  ],
  liver: [
    { name: 'Age', label: 'Age', type: 'number', min: 1, max: 120, step: 1, unit: 'years', hint: 'Patient age' },
    { name: 'Gender', label: 'Gender', type: 'select', options: [{ label: 'Male (1)', value: 1 }, { label: 'Female (0)', value: 0 }], hint: 'Biological gender' },
    { name: 'Total_Bilirubin', label: 'Total Bilirubin', type: 'number', min: 0.1, max: 80, step: 0.1, unit: 'mg/dl', hint: 'Total serum bilirubin' },
    { name: 'Direct_Bilirubin', label: 'Direct Bilirubin', type: 'number', min: 0.01, max: 30, step: 0.01, unit: 'mg/dl', hint: 'Conjugated bilirubin' },
    { name: 'Alkaline_Phosphotase', label: 'Alkaline Phosphatase', type: 'number', min: 10, max: 3000, step: 1, unit: 'IU/L', hint: 'ALP enzyme level' },
    { name: 'Alamine_Aminotransferase', label: 'Alamine Aminotransferase (ALT)', type: 'number', min: 5, max: 2000, step: 1, unit: 'IU/L', hint: 'SGPT / ALT enzyme' },
    { name: 'Aspartate_Aminotransferase', label: 'Aspartate Aminotransferase (AST)', type: 'number', min: 5, max: 2000, step: 1, unit: 'IU/L', hint: 'SGOT / AST enzyme' },
    { name: 'Total_Protiens', label: 'Total Proteins', type: 'number', min: 2, max: 15, step: 0.1, unit: 'g/dl', hint: 'Serum protein level' },
    { name: 'Albumin', label: 'Albumin', type: 'number', min: 1, max: 10, step: 0.1, unit: 'g/dl', hint: 'Serum albumin level' },
    { name: 'Albumin_and_Globulin_Ratio', label: 'Albumin/Globulin Ratio', type: 'number', min: 0.1, max: 5, step: 0.01, hint: 'A/G ratio' }
  ]
};

export default function AssessmentForm({ onPredictionComplete }) {
  const [condition, setCondition] = useState('heart');
  const [formData, setFormData] = useState(DEFAULT_VALS['heart']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleConditionChange = (newCond) => {
    setCondition(newCond);
    setFormData(DEFAULT_VALS[newCond]);
    setError(null);
  };

  const handleInputChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await predictRisk(condition, formData);
      onPredictionComplete(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Patient Risk Assessment Form</h2>
      </div>

      <div className="tab-group">
        <button
          type="button"
          className={`tab-btn ${condition === 'heart' ? 'active' : ''}`}
          onClick={() => handleConditionChange('heart')}
        >
          Heart Disease
        </button>
        <button
          type="button"
          className={`tab-btn ${condition === 'diabetes' ? 'active' : ''}`}
          onClick={() => handleConditionChange('diabetes')}
        >
          Diabetes
        </button>
        <button
          type="button"
          className={`tab-btn ${condition === 'liver' ? 'active' : ''}`}
          onClick={() => handleConditionChange('liver')}
        >
          Liver Disease
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {FIELD_CONFIGS[condition].map(field => (
            <div key={field.name} className="form-group">
              <label className="form-label">
                {field.label} {field.unit && <span style={{ fontWeight: 400, color: '#64748b' }}>({field.unit})</span>}
              </label>
              {field.type === 'select' ? (
                <select
                  className="form-select"
                  value={formData[field.name] ?? ''}
                  onChange={e => handleInputChange(field.name, e.target.value)}
                >
                  {field.options.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  className="form-input"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={formData[field.name] ?? ''}
                  onChange={e => handleInputChange(field.name, e.target.value)}
                  required
                />
              )}
              {field.hint && <span className="form-hint">{field.hint}</span>}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Evaluating Model...' : 'Calculate Risk Estimate'}
          </button>
          <button
            type="button"
            className="tab-btn"
            style={{ marginTop: '16px' }}
            onClick={() => setFormData(DEFAULT_VALS[condition])}
          >
            Reset Defaults
          </button>
        </div>
      </form>
    </div>
  );
}
