import React, { useState } from 'react';
import AssessmentForm from './components/AssessmentForm';
import ResultView from './components/ResultView';
import ModelComparison from './components/ModelComparison';
import HistoryView from './components/HistoryView';

export default function App() {
  const [activeTab, setActiveTab] = useState('assessment');
  const [currentResult, setCurrentResult] = useState(null);

  const handlePredictionComplete = (result) => {
    setCurrentResult(result);
    setActiveTab('result');
  };

  return (
    <div>
      <header className="navbar">
        <div className="brand-title">MediBuddy - Clinical Decision Support</div>
        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'assessment' ? 'active' : ''}`}
            onClick={() => setActiveTab('assessment')}
          >
            Risk Assessment Form
          </button>
          {currentResult && (
            <button
              className={`nav-btn ${activeTab === 'result' ? 'active' : ''}`}
              onClick={() => setActiveTab('result')}
            >
              Latest Result
            </button>
          )}
          <button
            className={`nav-btn ${activeTab === 'compare' ? 'active' : ''}`}
            onClick={() => setActiveTab('compare')}
          >
            Model Comparison
          </button>
          <button
            className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            Prediction History
          </button>
        </nav>
      </header>

      <main className="container">
        {activeTab === 'assessment' && (
          <AssessmentForm onPredictionComplete={handlePredictionComplete} />
        )}
        {activeTab === 'result' && (
          <ResultView result={currentResult} onBack={() => setActiveTab('assessment')} />
        )}
        {activeTab === 'compare' && (
          <ModelComparison />
        )}
        {activeTab === 'history' && (
          <HistoryView />
        )}
      </main>
    </div>
  );
}
