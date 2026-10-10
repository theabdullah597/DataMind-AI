'use client';
import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Legend, Cell } from 'recharts';

export default function Home() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [profile, setProfile] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  
  const [edaResults, setEdaResults] = useState(null);
  const [mlResults, setMlResults] = useState(null);
  const [explainResults, setExplainResults] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);
  
  const [loadingTask, setLoadingTask] = useState('');
  const [error, setError] = useState('');

  // Chat Agent State
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: "Hello! I'm your DataMind Agent. How can I help you analyze your data?" }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setUploading(true);
    setError('');
    setEdaResults(null);
    setMlResults(null);
    setExplainResults(null);
    setPredictionResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/datasets/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await response.json();
      setProfile(data.profile);
      setDatasetId(data.saved_filename);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleRunEDA = async () => {
    setLoadingTask('eda');
    setError('');
    try {
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}/eda`, {
        method: 'POST'
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'EDA failed');
      }
      const data = await response.json();
      setEdaResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingTask('');
    }
  };

  const handleTrainModel = async () => {
    const target = profile.dataset.column_names[profile.dataset.column_names.length - 1];
    setLoadingTask('ml');
    setError('');
    try {
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}/ml/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_column: target, task_type: "auto" })
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'ML failed');
      }
      const data = await response.json();
      setMlResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingTask('');
    }
  };

  const handleExplainModel = async () => {
    if (!mlResults || !mlResults.model_id) return;
    setLoadingTask('explain');
    setError('');
    try {
      const response = await fetch(`http://localhost:8000/models/${mlResults.model_id}/explain`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Explain failed');
      }
      const data = await response.json();
      setExplainResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingTask('');
    }
  };

  const handlePredictBatch = async (e) => {
    if (!e.target.files || !e.target.files.length) return;
    const predictFile = e.target.files[0];
    if (!mlResults || !mlResults.model_id) return;
    
    setLoadingTask('predict');
    setError('');
    const formData = new FormData();
    formData.append('file', predictFile);
    try {
      const response = await fetch(`http://localhost:8000/models/${mlResults.model_id}/predict/batch`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      const data = await response.json();
      setPredictionResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingTask('');
    }
  };

  const handleDownloadDataset = () => {
    window.open(`http://localhost:8000/datasets/${datasetId}/download`, '_blank');
  };

  const handleDownloadModel = () => {
    window.open(`http://localhost:8000/models/${mlResults.model_id}/download`, '_blank');
  };

  const handleDownloadPredictions = () => {
    if (!predictionResults || !predictionResults.predictions.length) return;
    const cols = Object.keys(predictionResults.predictions[0]);
    const csvContent = [
      cols.join(','),
      ...predictionResults.predictions.map(row => cols.map(c => row[c]).join(','))
    ].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${mlResults.model_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    const newMessages = [...chatMessages, { role: 'user', content: chatInput }];
    setChatMessages(newMessages);
    setChatInput('');
    setIsChatting(true);
    
    try {
      const response = await fetch('http://localhost:8000/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages, dataset_id: datasetId })
      });
      if (!response.ok) throw new Error('Chat failed');
      const data = await response.json();
      setChatMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setIsChatting(false);
    }
  };

  const handleReset = () => {
    setProfile(null);
    setDatasetId(null);
    setEdaResults(null);
    setMlResults(null);
    setExplainResults(null);
    setPredictionResults(null);
    setFile(null);
  };

  return (
    <div className="container" style={{ paddingBottom: '100px' }}>
      <header className="header animate-fade-in">
        <div className="logo">DataMind AI</div>
        <nav>
          <button className="btn btn-secondary">Documentation</button>
        </nav>
      </header>

      <main>
        {!profile ? (
          <div className="hero animate-fade-in">
            <h1 className="hero-title">
              Turn your data into <span>intelligence</span>
            </h1>
            <p className="hero-subtitle">
              Upload your dataset and instantly get AI-powered profiling, automatic cleaning, and machine learning models in seconds.
            </p>

            <div className="glass-card" style={{ maxWidth: '600px', margin: '0 auto' }}>
              <h2>Upload Dataset</h2>
              <div 
                className="file-upload" 
                onClick={() => document.getElementById('fileInput').click()}
              >
                <div className="file-upload-icon">📁</div>
                <h3>{file ? file.name : 'Click to select CSV or Excel file'}</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                  Max size: 50MB
                </p>
                <input 
                  type="file" 
                  id="fileInput" 
                  style={{ display: 'none' }} 
                  accept=".csv, .xlsx"
                  onChange={handleFileChange}
                />
              </div>

              {error && <div className="error-message">{error}</div>}

              <button 
                className="btn btn-primary" 
                style={{ width: '100%', margin: '1rem 0' }}
                onClick={handleUpload}
                disabled={uploading}
              >
                {uploading ? 'Analyzing...' : 'Analyze Dataset'}
              </button>
            </div>
            
            <div className="features-grid">
                <div className="glass-card">
                    <h3 style={{ marginBottom: '1rem' }}>Auto-Profiling</h3>
                    <p style={{ color: 'var(--text-secondary)' }}>Instantly discover missing values, data types, and high-level statistics without writing code.</p>
                </div>
                <div className="glass-card">
                    <h3 style={{ marginBottom: '1rem' }}>Smart Cleaning</h3>
                    <p style={{ color: 'var(--text-secondary)' }}>Easily manage duplicates, impute missing values using standard strategies, and cast data types.</p>
                </div>
                <div className="glass-card">
                    <h3 style={{ marginBottom: '1rem' }}>AutoML</h3>
                    <p style={{ color: 'var(--text-secondary)' }}>Train multiple machine learning models with a single click and pick the best performing one.</p>
                </div>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in">
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
              <button className="btn btn-secondary" onClick={handleReset}>Upload New</button>
              <button className="btn btn-primary" onClick={handleRunEDA} disabled={loadingTask === 'eda'}>
                {loadingTask === 'eda' ? 'Running EDA...' : 'Run Full EDA'}
              </button>
              <button className="btn btn-primary" onClick={handleTrainModel} disabled={loadingTask === 'ml'}>
                {loadingTask === 'ml' ? 'Training...' : 'Auto-Train ML (Last Column)'}
              </button>
              
              {mlResults && (
                <>
                  <button className="btn btn-secondary" onClick={handleExplainModel} disabled={loadingTask === 'explain'}>
                    {loadingTask === 'explain' ? 'Explaining...' : 'Explain Best Model'}
                  </button>
                  <div style={{ position: 'relative' }}>
                    <input type="file" id="predictInput" style={{ display: 'none' }} accept=".csv" onChange={handlePredictBatch} />
                    <button className="btn btn-secondary" onClick={() => document.getElementById('predictInput').click()} disabled={loadingTask === 'predict'}>
                      {loadingTask === 'predict' ? 'Predicting...' : 'Batch Predict (Upload CSV)'}
                    </button>
                  </div>
                </>
              )}
            </div>

            {error && <div className="error-message">{error}</div>}

            {edaResults && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2rem' }}>
                <h2 style={{ marginBottom: '1rem', color: 'var(--secondary)' }}>EDA Insights</h2>
                <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.8' }}>
                  {edaResults.insights?.map((insight, i) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
                <h3 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Numerical Distributions</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
                  {Object.entries(edaResults.numerical_analysis || {}).map(([col, stats]) => (
                    <div key={col} style={{ padding: '1rem', background: 'var(--bg-color)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>{col}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                        <span>Mean: {stats.mean}</span>
                        <span>Std: {stats.std}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                        <span>Min: {stats.min}</span>
                        <span>Max: {stats.max}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span>Q1: {stats.q1}</span>
                        <span>Q3: {stats.q3}</span>
                      </div>
                      <div style={{ marginTop: '1rem', height: '8px', background: 'var(--surface-color)', borderRadius: '4px', position: 'relative' }}>
                        <div style={{ position: 'absolute', left: `${Math.max(0, Math.min(100, ((stats.mean - stats.min)/(stats.max - stats.min || 1)) * 100))}%`, top: '-4px', width: '16px', height: '16px', borderRadius: '50%', background: 'var(--primary)', transform: 'translateX(-50%)', border: '2px solid white' }}></div>
                      </div>
                    </div>
                  ))}
                </div>

                {edaResults.categorical_analysis && Object.keys(edaResults.categorical_analysis).length > 0 && (
                  <>
                    <h3 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Categorical Distributions</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: '1rem' }}>
                      {Object.entries(edaResults.categorical_analysis).map(([col, data]) => {
                        const chartData = Object.entries(data.frequencies || {}).map(([name, val]) => ({ name, value: val }));
                        return (
                          <div key={col} style={{ padding: '1rem', background: 'var(--bg-color)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                            <div style={{ fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>{col}</div>
                            <div style={{ height: '250px' }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 20, left: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                                  <XAxis dataKey="name" stroke="var(--text-secondary)" tick={{fontSize: 12}} angle={-45} textAnchor="end" />
                                  <YAxis stroke="var(--text-secondary)" />
                                  <RechartsTooltip contentStyle={{ backgroundColor: 'var(--surface-color)', borderColor: 'var(--border-color)', color: 'white' }} />
                                  <Bar dataKey="value" fill="var(--primary)" />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}

                {profile?.missing_values?.counts && Object.values(profile.missing_values.counts).some(v => v > 0) && (
                  <>
                    <h3 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Missing Values Overview</h3>
                    <div style={{ height: '300px', background: 'var(--bg-color)', padding: '1rem', borderRadius: '8px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.entries(profile.missing_values.counts).map(([name, count]) => ({ name, count })).filter(d => d.count > 0)}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                          <XAxis dataKey="name" stroke="var(--text-secondary)" />
                          <YAxis stroke="var(--text-secondary)" />
                          <RechartsTooltip contentStyle={{ backgroundColor: 'var(--surface-color)', borderColor: 'var(--border-color)', color: 'white' }} />
                          <Bar dataKey="count" fill="var(--error)" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                )}
              </div>
            )}

            {mlResults && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h2 style={{ color: 'var(--secondary)', margin: 0 }}>AutoML Results</h2>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <button className="btn btn-secondary" onClick={handleDownloadDataset}>Download Cleaned Dataset</button>
                        <button className="btn btn-primary" onClick={handleDownloadModel}>Download Model (.joblib)</button>
                    </div>
                </div>
                
                <div className="stat-card" style={{ marginBottom: '1rem' }}>
                  <div className="stat-value" style={{ fontSize: '1.5rem' }}>Best Model: {mlResults.model_name}</div>
                  <div className="stat-label">Task Type: {mlResults.task_type}</div>
                  <div className="stat-label" style={{ marginTop: '0.5rem', color: 'var(--success)' }}>
                    {mlResults.recommendation_reason}
                  </div>
                </div>

                <h3 style={{ marginBottom: '1rem' }}>Model Comparison</h3>
                <div style={{ overflowX: 'auto', marginBottom: '2rem' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'var(--bg-color)', borderRadius: '8px', overflow: 'hidden' }}>
                    <thead>
                      <tr style={{ background: 'var(--surface-color)', borderBottom: '1px solid var(--border-color)' }}>
                        <th style={{ padding: '1rem' }}>Model</th>
                        {Object.keys(Object.values(mlResults.all_results)[0]).filter(k => k !== 'confusion_matrix' && k !== 'plot_data').map(k => (
                          <th key={k} style={{ padding: '1rem', textTransform: 'capitalize' }}>{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(mlResults.all_results).map(([modelName, metrics]) => (
                        <tr key={modelName} style={{ borderBottom: '1px solid var(--border-color)', background: modelName === mlResults.model_name ? 'rgba(76, 175, 80, 0.1)' : 'transparent' }}>
                          <td style={{ padding: '1rem', fontWeight: 'bold' }}>{modelName} {modelName === mlResults.model_name && '🏆'}</td>
                          {Object.entries(metrics).filter(([k]) => k !== 'confusion_matrix' && k !== 'plot_data').map(([k, v]) => (
                            <td key={k} style={{ padding: '1rem' }}>{v}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {mlResults.metrics.confusion_matrix && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Confusion Matrix (Best Model)</h3>
                    <div style={{ display: 'inline-block', background: 'var(--bg-color)', padding: '1rem', borderRadius: '8px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {mlResults.metrics.confusion_matrix.map((row, i) => (
                          <div key={i} style={{ display: 'flex', gap: '4px' }}>
                            {row.map((cell, j) => {
                              const maxVal = Math.max(...mlResults.metrics.confusion_matrix.flat());
                              const intensity = Math.min(1, Math.max(0.1, cell / (maxVal || 1)));
                              return (
                                <div key={j} style={{ 
                                  width: '50px', height: '50px', 
                                  display: 'flex', alignItems: 'center', justifyContent: 'center', 
                                  background: `rgba(76, 175, 80, ${intensity})`,
                                  color: intensity > 0.5 ? 'black' : 'white',
                                  fontWeight: 'bold', borderRadius: '4px' 
                                }}>
                                  {cell}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                      <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Rows: Actual, Columns: Predicted</p>
                    </div>
                  </div>
                )}

                {mlResults.metrics.plot_data && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Actual vs Predicted (Best Model)</h3>
                    <div style={{ height: '300px', width: '100%', background: 'var(--bg-color)', padding: '1rem', borderRadius: '8px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                          <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--text-secondary)" />
                          <YAxis type="number" dataKey="predicted" name="Predicted" stroke="var(--text-secondary)" />
                          <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--surface-color)', borderColor: 'var(--border-color)', color: 'white' }} />
                          <Scatter name="Predictions" data={mlResults.metrics.plot_data} fill="var(--primary)" />
                          <ReferenceLine segment={[{ x: 'dataMin', y: 'dataMin' }, { x: 'dataMax', y: 'dataMax' }]} stroke="var(--success)" strokeDasharray="3 3" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

              </div>
            )}

            {explainResults && explainResults.explanations && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2rem' }}>
                <h2 style={{ marginBottom: '1rem', color: 'var(--secondary)' }}>Model Explainability</h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontStyle: 'italic' }}>
                  {explainResults.explanations.notice}
                </p>
                {explainResults.explanations.feature_importances && (
                  <div>
                    <h3 style={{ marginBottom: '1rem' }}>Feature Importances</h3>
                    <div style={{ display: 'grid', gap: '0.5rem' }}>
                      {Object.entries(explainResults.explanations.feature_importances)
                        .sort((a, b) => b[1] - a[1])
                        .map(([feature, importance]) => (
                          <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '150px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{feature}</div>
                            <div style={{ flex: 1, background: 'var(--bg-color)', height: '12px', borderRadius: '6px', overflow: 'hidden' }}>
                              <div style={{ width: `${Math.min(100, Math.max(0, importance * 100))}%`, height: '100%', background: 'var(--primary)' }}></div>
                            </div>
                            <div style={{ width: '50px', textAlign: 'right', fontSize: '0.875rem' }}>{(importance * 100).toFixed(1)}%</div>
                          </div>
                      ))}
                    </div>
                  </div>
                )}
                {explainResults.explanations.coefficients && (
                  <div>
                    <h3 style={{ marginBottom: '1rem' }}>Coefficients</h3>
                    <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.8' }}>
                      {Object.entries(explainResults.explanations.coefficients).map(([feature, coef]) => (
                        <li key={feature}>{feature}: {coef.toFixed(4)}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {predictionResults && predictionResults.predictions && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h2 style={{ color: 'var(--secondary)', margin: 0 }}>Batch Predictions</h2>
                    <button className="btn btn-primary" onClick={handleDownloadPredictions}>Download Predictions (CSV)</button>
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                        {Object.keys(predictionResults.predictions[0]).map(col => (
                          <th key={col} style={{ padding: '1rem', color: 'var(--text-secondary)' }}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {predictionResults.predictions.slice(0, 10).map((row, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          {Object.values(row).map((val, j) => (
                            <td key={j} style={{ padding: '1rem' }}>{val}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {predictionResults.predictions.length > 10 && (
                    <p style={{ marginTop: '1rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
                      Showing first 10 of {predictionResults.predictions.length} predictions.
                    </p>
                  )}
                </div>
              </div>
            )}

            <h2>Dataset Profile</h2>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{profile.dataset?.rows}</div>
                <div className="stat-label">Total Rows</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{profile.dataset?.columns}</div>
                <div className="stat-label">Total Columns</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{profile.duplicates?.count}</div>
                <div className="stat-label">Duplicate Rows</div>
              </div>
            </div>
            
            <div className="glass-card" style={{ marginTop: '2rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>Data Quality</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '1rem'}}>
                  {profile.missing_values?.counts && Object.keys(profile.missing_values.counts).map(col => (
                    <div key={col} style={{ padding: '1rem', background: 'var(--surface-color)', borderRadius: '8px', border: '1px solid var(--border-color)'}}>
                      <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>{col}</div>
                      <div style={{ fontSize: '0.875rem', color: profile.missing_values.counts[col] > 0 ? 'var(--error)' : 'var(--success)'}}>
                        {profile.missing_values.counts[col]} missing ({profile.missing_values.percentages[col]?.toFixed(1)}%)
                      </div>
                      <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        Type: {profile.data_types?.[col]}
                      </div>
                    </div>
                  ))}
              </div>
            </div>

          </div>
        )}
      </main>

      {profile && (
        <>
          <button 
            onClick={() => setIsChatOpen(!isChatOpen)}
            style={{ position: 'fixed', bottom: '20px', right: '20px', width: '60px', height: '60px', borderRadius: '30px', background: 'var(--primary)', color: 'white', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.3)', zIndex: 1001, cursor: 'pointer', fontSize: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'transform 0.2s' }}
          >
            {isChatOpen ? '✕' : '💬'}
          </button>
          
          {isChatOpen && (
            <div className="chat-panel animate-fade-in" style={{ position: 'fixed', bottom: '90px', right: '20px', width: '350px', background: 'var(--surface-color)', border: '1px solid var(--border-color)', borderRadius: '12px', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 10px 40px rgba(0,0,0,0.5)', zIndex: 1000 }}>
              <div style={{ padding: '1rem', background: 'var(--primary)', color: 'white', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between' }}>
                <span>DataMind AI Agent</span>
                <span>✨</span>
              </div>
              <div style={{ height: '350px', overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {chatMessages.map((msg, i) => (
                  <div key={i} style={{ 
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', 
                    background: msg.role === 'user' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', 
                    border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
                    padding: '0.75rem 1rem', 
                    borderRadius: '12px', 
                    borderBottomRightRadius: msg.role === 'user' ? '0' : '12px',
                    borderBottomLeftRadius: msg.role === 'assistant' ? '0' : '12px',
                    maxWidth: '85%', 
                    fontSize: '0.875rem',
                    lineHeight: '1.5',
                    wordBreak: 'break-word'
                  }}>
                    {msg.content}
                  </div>
                ))}
                {isChatting && <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', padding: '0.5rem' }}>Agent is thinking and running tools...</div>}
              </div>
              <form onSubmit={handleSendMessage} style={{ display: 'flex', borderTop: '1px solid var(--border-color)', background: 'var(--bg-color)' }}>
                <input 
                  type="text" 
                  value={chatInput} 
                  onChange={e => setChatInput(e.target.value)} 
                  placeholder="E.g., Clean the dataset and analyze it..." 
                  style={{ flex: 1, background: 'transparent', border: 'none', padding: '1rem', color: 'white', outline: 'none', fontSize: '0.875rem' }} 
                  disabled={isChatting}
                />
                <button type="submit" className="btn btn-primary" style={{ borderRadius: 0, padding: '0 1rem' }} disabled={isChatting}>
                  Send
                </button>
              </form>
            </div>
          )}
        </>
      )}
    </div>
  );
}
