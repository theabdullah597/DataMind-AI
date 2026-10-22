'use client';
import { useState, useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Legend, Cell
} from 'recharts';

export default function Home() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [profile, setProfile] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState('');

  const [edaResults, setEdaResults] = useState(null);
  const [mlResults, setMlResults] = useState(null);
  const [explainResults, setExplainResults] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);

  const [loadingTask, setLoadingTask] = useState('');
  const [error, setError] = useState('');
  const [edaInsightFilter, setEdaInsightFilter] = useState('ALL');
  const [predictSearchQuery, setPredictSearchQuery] = useState('');

  // Documentation Modal State
  const [isDocModalOpen, setIsDocModalOpen] = useState(false);
  const [activeDocTab, setActiveDocTab] = useState('quickstart');

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
      // Default selected target to the last column
      if (data.profile?.dataset?.column_names?.length) {
        const cols = data.profile.dataset.column_names;
        setSelectedTarget(cols[cols.length - 1]);
      }
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
      const targetQuery = selectedTarget ? `?target_column=${encodeURIComponent(selectedTarget)}` : '';
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}/eda${targetQuery}`, {
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
    const target = selectedTarget || profile.dataset.column_names[profile.dataset.column_names.length - 1];
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
      // reset file input
      e.target.value = '';
    }
  };

  const handleDownloadTemplate = () => {
    if (!mlResults || !mlResults.model_id) return;
    window.open(`http://localhost:8000/models/${mlResults.model_id}/template`, '_blank');
  };

  const handleDownloadDataset = () => {
    window.open(`http://localhost:8000/datasets/${datasetId}/download`, '_blank');
  };

  const handleDownloadModel = () => {
    window.open(`http://localhost:8000/models/${mlResults.model_id}/download`, '_blank');
  };

  const handleDownloadPredictions = () => {
    if (!predictionResults || !predictionResults.predictions?.length) return;
    const cols = Object.keys(predictionResults.predictions[0]);
    const csvContent = [
      cols.join(','),
      ...predictionResults.predictions.map(row => cols.map(c => `"${row[c] !== undefined ? row[c] : ''}"`).join(','))
    ].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${mlResults.model_id.slice(0, 8)}.csv`;
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
    setSelectedTarget('');
    setError('');
  };

  // Normalize insights for display
  const normalizedInsights = useMemo(() => {
    if (!edaResults?.insights) return [];
    return edaResults.insights.map((item, idx) => {
      if (typeof item === 'string') {
        return {
          id: idx,
          category: 'General',
          badge: 'Info',
          title: item,
          description: item,
          recommendation: null,
          technical_detail: null
        };
      }
      return {
        id: idx,
        category: item.category || 'General',
        badge: item.badge || 'Info',
        title: item.title || item.text || 'Insight',
        description: item.description || item.text || '',
        recommendation: item.recommendation || null,
        technical_detail: item.technical_detail || null
      };
    });
  }, [edaResults]);

  const filteredInsights = useMemo(() => {
    if (edaInsightFilter === 'ALL') return normalizedInsights;
    return normalizedInsights.filter(ins => ins.category?.toLowerCase().includes(edaInsightFilter.toLowerCase()));
  }, [normalizedInsights, edaInsightFilter]);

  // Overfitting / Underfitting Comparison Chart Data
  const modelFitChartData = useMemo(() => {
    if (!mlResults?.all_results) return [];
    const isClf = mlResults.task_type === 'classification';

    return Object.entries(mlResults.all_results).map(([name, metrics]) => {
      let trainScore = 0;
      let testScore = 0;
      let scoreName = isClf ? 'Accuracy' : 'R² Score';

      if (isClf) {
        trainScore = (metrics.train_accuracy || 0) * 100;
        testScore = (metrics.test_accuracy || metrics.accuracy || 0) * 100;
      } else {
        trainScore = (metrics.train_r2 || 0) * 100;
        testScore = (metrics.test_r2 || metrics.r2_score || 0) * 100;
      }

      const diag = metrics.fit_diagnosis || {};
      return {
        model: name,
        shortName: name.replace(' Classifier', '').replace(' Regressor', ''),
        trainScore: Math.round(trainScore * 10) / 10,
        testScore: Math.round(testScore * 10) / 10,
        gap: Math.round(Math.abs(trainScore - testScore) * 10) / 10,
        status: diag.status || (trainScore - testScore > 12 ? 'Overfitting' : testScore < 40 ? 'Underfitting' : 'Good Fit'),
        badge: diag.badge || diag.status || 'Evaluated',
        action: diag.action || '',
        scoreName
      };
    });
  }, [mlResults]);

  // Filtered batch predictions
  const filteredPredictions = useMemo(() => {
    if (!predictionResults?.predictions) return [];
    if (!predictSearchQuery.trim()) return predictionResults.predictions;
    const q = predictSearchQuery.toLowerCase();
    return predictionResults.predictions.filter(row =>
      Object.values(row).some(val => String(val).toLowerCase().includes(q))
    );
  }, [predictionResults, predictSearchQuery]);

  return (
    <div className="container" style={{ paddingBottom: '100px' }}>
      {/* Top Bar Header */}
      <header className="header animate-fade-in">
        <div className="logo" style={{ cursor: 'pointer' }} onClick={handleReset}>
          DataMind AI <span style={{ fontSize: '0.8rem', background: 'var(--primary)', padding: '2px 8px', borderRadius: '6px', color: 'white', verticalAlign: 'middle', marginLeft: '6px' }}>PRO</span>
        </div>
        <nav style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            className="btn btn-secondary"
            onClick={() => setIsDocModalOpen(true)}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <span>📖</span> Documentation & Guide
          </button>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none' }}
          >
            <span>⚡</span> API Swagger
          </a>
        </nav>
      </header>

      {/* Main Content */}
      <main>
        {!profile ? (
          <div className="hero animate-fade-in">
            <h1 className="hero-title">
              Turn your data into <span>intelligence</span>
            </h1>
            <p className="hero-subtitle">
              Upload your dataset and instantly get plain-English exploratory analysis, transparent charts, automated machine learning models, and real-time overfitting diagnostics.
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
                  Supports .csv and .xlsx (Max size: 50MB)
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
                {uploading ? 'Analyzing Dataset...' : 'Analyze Dataset Now'}
              </button>
            </div>

            <div className="features-grid">
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>📊</span> Understandable EDA
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Clear, jargon-free distribution histograms and automated plain-English data insights that anyone can interpret.
                </p>
              </div>
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>🧠</span> Overfitting Detection
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Automatic Train vs Test performance comparisons so you immediately know if your model overfits, underfits, or generalizes well.
                </p>
              </div>
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>🔮</span> Batch Predictions
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Upload new unlabelled CSV files to run bulk predictions across hundreds of rows in seconds, with downloadable results.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in">
            {/* Action Bar */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap', alignItems: 'center', background: 'var(--surface-color)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <button className="btn btn-secondary" onClick={handleReset}>Upload New Dataset</button>

              {/* Target column selector */}
              {profile.dataset?.column_names && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Target Column:</label>
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    style={{ background: 'var(--bg-color)', color: 'white', border: '1px solid var(--border-color)', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.9rem' }}
                  >
                    {profile.dataset.column_names.map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
              )}

              <button className="btn btn-primary" onClick={handleRunEDA} disabled={loadingTask === 'eda'}>
                {loadingTask === 'eda' ? 'Generating EDA...' : '📊 Run Plain-English EDA'}
              </button>

              <button className="btn btn-primary" onClick={handleTrainModel} disabled={loadingTask === 'ml'}>
                {loadingTask === 'ml' ? 'Training AutoML...' : `🚀 Train Models on "${selectedTarget || 'Target'}"`}
              </button>

              {mlResults && (
                <>
                  <button className="btn btn-secondary" onClick={handleExplainModel} disabled={loadingTask === 'explain'}>
                    {loadingTask === 'explain' ? 'Explaining...' : '💡 Explain Features'}
                  </button>
                  <button className="btn btn-secondary" onClick={handleDownloadTemplate}>
                    📥 Download Batch Template
                  </button>
                  <div style={{ position: 'relative' }}>
                    <input type="file" id="predictInput" style={{ display: 'none' }} accept=".csv, .xlsx" onChange={handlePredictBatch} />
                    <button
                      className="btn btn-secondary"
                      onClick={() => document.getElementById('predictInput').click()}
                      disabled={loadingTask === 'predict'}
                      style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'var(--primary)', color: '#a5b4fc' }}
                    >
                      {loadingTask === 'predict' ? 'Predicting Rows...' : '🔮 Batch Predict (Upload CSV)'}
                    </button>
                  </div>
                </>
              )}
            </div>

            {error && <div className="error-message">{error}</div>}

            {/* EDA Results Section */}
            {edaResults && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                  <div>
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>📊</span> Plain-English EDA Insights
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
                      Automated discoveries, data patterns, and actionable recommendations anyone can understand.
                    </p>
                  </div>
                  {/* Category filters */}
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {['ALL', 'Target', 'Distribution', 'Correlation', 'Quality'].map(cat => (
                      <button
                        key={cat}
                        onClick={() => setEdaInsightFilter(cat)}
                        style={{
                          padding: '0.35rem 0.75rem',
                          borderRadius: '20px',
                          border: '1px solid var(--border-color)',
                          background: edaInsightFilter === cat ? 'var(--primary)' : 'transparent',
                          color: edaInsightFilter === cat ? 'white' : 'var(--text-secondary)',
                          fontSize: '0.8rem',
                          cursor: 'pointer',
                          fontWeight: '600'
                        }}
                      >
                        {cat === 'ALL' ? `All (${normalizedInsights.length})` : cat}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Insight Cards Grid */}
                <div className="insight-grid" style={{ marginBottom: '2.5rem' }}>
                  {filteredInsights.map((ins) => (
                    <div key={ins.id} className="insight-card">
                      <div className="insight-header">
                        <span className="insight-title">{ins.title}</span>
                        <span className={`badge badge-${ins.badge.toLowerCase()}`}>
                          {ins.badge}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                        {ins.description}
                      </p>
                      {ins.recommendation && (
                        <div style={{ background: 'rgba(16, 185, 129, 0.08)', borderLeft: '3px solid var(--success)', padding: '0.5rem 0.75rem', borderRadius: '0 6px 6px 0', fontSize: '0.8rem', color: '#a7f3d0' }}>
                          <strong>💡 Action:</strong> {ins.recommendation}
                        </div>
                      )}
                      {ins.technical_detail && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.5rem' }}>
                          <span style={{ opacity: 0.8 }}>Stat: {ins.technical_detail}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Numerical Distributions Section */}
                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '2rem', marginBottom: '2.5rem' }}>
                  <div style={{ marginBottom: '1.25rem' }}>
                    <h3 style={{ fontSize: '1.3rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>📈</span> Numerical Distributions & Histograms
                    </h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                      Real histograms showing value frequencies and distribution shapes. Hover over bars to see exact counts.
                    </p>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1.5rem' }}>
                    {Object.entries(edaResults.numerical_analysis || {}).map(([col, stats]) => (
                      <div key={col} className="hist-card">
                        <div className="hist-header">
                          <div>
                            <div className="hist-title">{col}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                              {stats.count?.toLocaleString()} rows • Min: {stats.min} | Max: {stats.max}
                            </div>
                          </div>
                          {stats.shape && (
                            <span className={`badge ${stats.shape.includes('Normal') ? 'badge-tip' : 'badge-info'}`}>
                              {stats.shape}
                            </span>
                          )}
                        </div>

                        {/* Stat pills */}
                        <div className="hist-pills">
                          <div className="stat-pill"><span className="stat-pill-label">Mean:</span><span className="stat-pill-val">{stats.mean}</span></div>
                          <div className="stat-pill"><span className="stat-pill-label">Median:</span><span className="stat-pill-val">{stats.median}</span></div>
                          <div className="stat-pill"><span className="stat-pill-label">Std:</span><span className="stat-pill-val">{stats.std}</span></div>
                          <div className="stat-pill"><span className="stat-pill-label">Q1:</span><span className="stat-pill-val">{stats.q1}</span></div>
                          <div className="stat-pill"><span className="stat-pill-label">Q3:</span><span className="stat-pill-val">{stats.q3}</span></div>
                        </div>

                        {/* Real Recharts Histogram */}
                        <div style={{ height: '220px', width: '100%', marginTop: '0.5rem' }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={stats.histogram || []} margin={{ top: 10, right: 10, bottom: 25, left: -10 }}>
                              <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                              <XAxis
                                dataKey="bin"
                                stroke="var(--text-secondary)"
                                tick={{ fontSize: 10, fill: '#94a3b8' }}
                                angle={-25}
                                textAnchor="end"
                                interval={0}
                              />
                              <YAxis
                                stroke="var(--text-secondary)"
                                tick={{ fontSize: 11, fill: '#94a3b8' }}
                                allowDecimals={false}
                              />
                              <RechartsTooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                      <div style={{ background: '#1e293b', border: '1px solid #475569', padding: '8px 12px', borderRadius: '8px', color: '#f8fafc', fontSize: '0.825rem', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}>
                                        <div style={{ fontWeight: 'bold', color: '#818cf8', marginBottom: '4px' }}>Range: {data.bin}</div>
                                        <div>Count: <span style={{ fontWeight: 'bold', color: 'white' }}>{data.count}</span> records</div>
                                        {data.percentage !== undefined && <div>Share: {data.percentage}% of data</div>}
                                      </div>
                                    );
                                  }
                                  return null;
                                }}
                              />
                              <Bar dataKey="count" fill="url(#histGradient)" radius={[4, 4, 0, 0]} />
                              <defs>
                                <linearGradient id="histGradient" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.9} />
                                  <stop offset="100%" stopColor="#4338ca" stopOpacity={0.6} />
                                </linearGradient>
                              </defs>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>

                        {/* Plain English explanation banner */}
                        <div className="hist-summary-banner">
                          <strong>💡 What this means:</strong> {stats.simple_summary || `${col} ranges from ${stats.min} to ${stats.max} with an average of ${stats.mean}.`}
                          {stats.shape_desc && <div style={{ marginTop: '3px', opacity: 0.9 }}>{stats.shape_desc}</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Categorical Distributions */}
                {edaResults.categorical_analysis && Object.keys(edaResults.categorical_analysis).length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '2rem', marginBottom: '2.5rem' }}>
                    <div style={{ marginBottom: '1.25rem' }}>
                      <h3 style={{ fontSize: '1.3rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>🏷️</span> Categorical Distributions
                      </h3>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                        Breakdown of unique categories, counts, and relative percentages.
                      </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: '1.5rem' }}>
                      {Object.entries(edaResults.categorical_analysis).map(([col, data]) => {
                        const chartData = Object.entries(data.frequencies || {}).map(([name, val]) => ({
                          name: String(name),
                          value: val,
                          pct: data.percentages?.[name] || 0
                        }));
                        return (
                          <div key={col} className="hist-card">
                            <div className="hist-header">
                              <div>
                                <div className="hist-title">{col}</div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                                  {data.unique_categories} unique categories • Most common: <strong style={{ color: 'var(--primary)' }}>{data.most_common}</strong> ({data.most_common_count})
                                </div>
                              </div>
                              <span className="badge badge-info">{data.unique_categories} Categories</span>
                            </div>

                            <div style={{ height: '220px', width: '100%', marginTop: '0.5rem' }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 25, left: -10 }}>
                                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                                  <XAxis
                                    dataKey="name"
                                    stroke="var(--text-secondary)"
                                    tick={{ fontSize: 11, fill: '#94a3b8' }}
                                    angle={-25}
                                    textAnchor="end"
                                  />
                                  <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 11, fill: '#94a3b8' }} allowDecimals={false} />
                                  <RechartsTooltip
                                    content={({ active, payload }) => {
                                      if (active && payload && payload.length) {
                                        const d = payload[0].payload;
                                        return (
                                          <div style={{ background: '#1e293b', border: '1px solid #475569', padding: '8px 12px', borderRadius: '8px', color: '#f8fafc', fontSize: '0.825rem' }}>
                                            <div style={{ fontWeight: 'bold', color: '#ec4899', marginBottom: '4px' }}>Category: {d.name}</div>
                                            <div>Count: <span style={{ fontWeight: 'bold' }}>{d.value}</span> rows</div>
                                            <div>Percentage: <span style={{ fontWeight: 'bold' }}>{d.pct}%</span></div>
                                          </div>
                                        );
                                      }
                                      return null;
                                    }}
                                  />
                                  <Bar dataKey="value" fill="url(#catGradient)" radius={[4, 4, 0, 0]} />
                                  <defs>
                                    <linearGradient id="catGradient" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="#ec4899" stopOpacity={0.9} />
                                      <stop offset="100%" stopColor="#be185d" stopOpacity={0.6} />
                                    </linearGradient>
                                  </defs>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Strong Correlations */}
                {edaResults.correlation_analysis?.strong_correlations?.length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '2rem', marginBottom: '1.5rem' }}>
                    <div style={{ marginBottom: '1.25rem' }}>
                      <h3 style={{ fontSize: '1.3rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>🔗</span> Strong Feature Relationships (Correlations)
                      </h3>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                        Columns that move together. High correlation (&gt; 0.70) indicates shared predictive information.
                      </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
                      {edaResults.correlation_analysis.strong_correlations.map((corr, idx) => (
                        <div key={idx} style={{ background: 'var(--bg-color)', border: '1px solid var(--border-color)', padding: '1rem', borderRadius: '10px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: '600', color: 'white' }}>{corr.column_1} ↔ {corr.column_2}</span>
                            <span className={`badge ${corr.correlation > 0 ? 'badge-tip' : 'badge-alert'}`}>
                              {corr.correlation > 0 ? '📈 Positive' : '📉 Negative'} {corr.correlation}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                            {corr.correlation > 0
                              ? `When ${corr.column_1} increases, ${corr.column_2} almost always increases as well.`
                              : `When ${corr.column_1} increases, ${corr.column_2} tends to decrease significantly.`}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* AutoML Results & Overfitting / Underfitting Diagnostics */}
            {mlResults && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                  <div>
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>🚀</span> AutoML Results & Training Diagnostics
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
                      Models trained and evaluated against unseen test records with automated overfitting detection.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={handleDownloadDataset}>Download Cleaned Data</button>
                    <button className="btn btn-primary" onClick={handleDownloadModel}>Download Best Model (.joblib)</button>
                  </div>
                </div>

                {/* Best Model Summary Card */}
                <div className="stat-card" style={{ marginBottom: '2rem', background: 'rgba(30, 41, 59, 0.8)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                      <div className="stat-value" style={{ fontSize: '1.6rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        🏆 Best Model: <span style={{ color: 'var(--primary)' }}>{mlResults.model_name}</span>
                      </div>
                      <div className="stat-label" style={{ fontSize: '0.95rem', marginTop: '0.5rem' }}>
                        Task: <strong style={{ textTransform: 'capitalize', color: 'white' }}>{mlResults.task_type}</strong> | Target: <strong style={{ color: 'var(--secondary)' }}>{mlResults.target_column || selectedTarget}</strong>
                      </div>
                      <div style={{ marginTop: '0.75rem', color: '#34d399', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>✓</span> {mlResults.recommendation_reason}
                      </div>
                    </div>

                    {/* Best Model Diagnostic Badge */}
                    {mlResults.metrics?.fit_diagnosis && (
                      <div style={{ background: 'var(--bg-color)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)', minWidth: '240px' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
                          Model Fit Diagnosis
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                          <span className={`badge ${
                            mlResults.metrics.fit_diagnosis.status === 'Good Fit' ? 'badge-good-fit' :
                            mlResults.metrics.fit_diagnosis.status === 'Overfitting' ? 'badge-overfitting' : 'badge-underfitting'
                          }`} style={{ fontSize: '0.85rem', padding: '0.35rem 0.75rem' }}>
                            {mlResults.metrics.fit_diagnosis.badge}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.8rem', color: '#cbd5e1', lineHeight: '1.4' }}>
                          {mlResults.metrics.fit_diagnosis.summary}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Overfitting / Underfitting Educational Explainer */}
                <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.5rem', marginBottom: '2rem' }}>
                  <h3 style={{ fontSize: '1.15rem', color: 'white', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span>🧠</span> How to Identify if a Model Overfits or Underfits
                  </h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem', lineHeight: '1.6' }}>
                    A good machine learning model must perform well on <strong>unseen test data</strong>, not just the training data it memorized. Check the Train vs Test scores below:
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#34d399', marginBottom: '0.25rem' }}>🟢 Good Fit (Balanced)</div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1' }}>
                        Train and Test scores are both strong and close together (Gap &lt; 10%). The model learned true patterns and generalizes reliably.
                      </div>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#f87171', marginBottom: '0.25rem' }}>🔴 Overfitting (High Variance)</div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1' }}>
                        Train score is high (e.g. 98%), but Test score is noticeably lower. The model memorized training noise like cramming for an exam.
                      </div>
                    </div>
                    <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#fbbf24', marginBottom: '0.25rem' }}>🟠 Underfitting (High Bias)</div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1' }}>
                        Both Train and Test scores are low. The model was too simplistic and failed to capture the relationships in the data.
                      </div>
                    </div>
                  </div>
                </div>

                {/* Train vs Test Visual Comparison Chart */}
                {modelFitChartData.length > 0 && (
                  <div style={{ marginBottom: '2rem', background: 'var(--bg-color)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.25rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                      <div>
                        <div style={{ fontWeight: 'bold', color: 'white', fontSize: '1.05rem' }}>
                          Train Score vs Test Score Comparison (Generalization Gap)
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          Higher test score is better. A small gap between blue and green bars indicates great generalization.
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><span style={{ width: '12px', height: '12px', background: '#818cf8', borderRadius: '2px' }}></span> Training Score</span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><span style={{ width: '12px', height: '12px', background: '#10b981', borderRadius: '2px' }}></span> Test Score (Unseen)</span>
                      </div>
                    </div>

                    <div style={{ height: '260px', width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={modelFitChartData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                          <XAxis dataKey="shortName" stroke="var(--text-secondary)" tick={{ fontSize: 12, fill: '#cbd5e1' }} />
                          <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 11, fill: '#94a3b8' }} unit="%" />
                          <RechartsTooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const d = payload[0].payload;
                                return (
                                  <div style={{ background: '#1e293b', border: '1px solid #475569', padding: '10px 14px', borderRadius: '8px', color: '#f8fafc', fontSize: '0.85rem' }}>
                                    <div style={{ fontWeight: 'bold', color: 'white', marginBottom: '6px' }}>{d.model}</div>
                                    <div style={{ color: '#818cf8' }}>Training: <span style={{ fontWeight: 'bold' }}>{d.trainScore}%</span></div>
                                    <div style={{ color: '#34d399' }}>Testing: <span style={{ fontWeight: 'bold' }}>{d.testScore}%</span></div>
                                    <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                      Generalization Gap: <strong style={{ color: d.gap > 12 ? '#f87171' : '#34d399' }}>{d.gap}%</strong>
                                    </div>
                                    <div style={{ marginTop: '4px' }}>
                                      Diagnosis: <span className={`badge ${d.status === 'Good Fit' ? 'badge-good-fit' : d.status === 'Overfitting' ? 'badge-overfitting' : 'badge-underfitting'}`}>{d.status}</span>
                                    </div>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Bar dataKey="trainScore" name="Training Score" fill="#818cf8" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="testScore" name="Test Score" fill="#10b981" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Model Comparison Table */}
                <h3 style={{ marginBottom: '1rem', color: 'white' }}>Detailed Model Comparison</h3>
                <div style={{ overflowX: 'auto', marginBottom: '2rem' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'var(--bg-color)', borderRadius: '8px', overflow: 'hidden' }}>
                    <thead>
                      <tr style={{ background: 'var(--surface-color)', borderBottom: '1px solid var(--border-color)' }}>
                        <th style={{ padding: '1rem' }}>Model</th>
                        <th style={{ padding: '1rem' }}>Fit Diagnosis</th>
                        <th style={{ padding: '1rem' }}>Train Score</th>
                        <th style={{ padding: '1rem' }}>Test Score</th>
                        <th style={{ padding: '1rem' }}>Gap</th>
                        {Object.keys(Object.values(mlResults.all_results)[0])
                          .filter(k => !['confusion_matrix', 'plot_data', 'fit_diagnosis', 'train_r2', 'train_rmse', 'train_mae', 'test_r2', 'test_rmse', 'train_accuracy', 'train_f1', 'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std', 'cv_r2_mean', 'cv_r2_std'].includes(k))
                          .map(k => (
                            <th key={k} style={{ padding: '1rem', textTransform: 'capitalize' }}>{k}</th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(mlResults.all_results).map(([modelName, metrics]) => {
                        const diag = metrics.fit_diagnosis || {};
                        const isBest = modelName === mlResults.model_name;
                        return (
                          <tr key={modelName} style={{ borderBottom: '1px solid var(--border-color)', background: isBest ? 'rgba(99, 102, 241, 0.08)' : 'transparent' }}>
                            <td style={{ padding: '1rem', fontWeight: 'bold' }}>
                              {modelName} {isBest && '🏆'}
                            </td>
                            <td style={{ padding: '1rem' }}>
                              <span className={`badge ${
                                diag.status === 'Good Fit' ? 'badge-good-fit' :
                                diag.status === 'Overfitting' ? 'badge-overfitting' :
                                diag.status === 'Underfitting' ? 'badge-underfitting' : 'badge-info'
                              }`}>
                                {diag.badge || diag.status || 'Evaluated'}
                              </span>
                            </td>
                            <td style={{ padding: '1rem', color: '#818cf8', fontWeight: '600' }}>
                              {metrics.train_accuracy !== undefined ? `${(metrics.train_accuracy * 100).toFixed(1)}%` : metrics.train_r2 !== undefined ? metrics.train_r2 : '-'}
                            </td>
                            <td style={{ padding: '1rem', color: '#34d399', fontWeight: '600' }}>
                              {metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}%` : metrics.r2_score !== undefined ? metrics.r2_score : metrics.rmse !== undefined ? metrics.rmse : '-'}
                            </td>
                            <td style={{ padding: '1rem', color: diag.generalization_gap > 0.12 ? '#f87171' : '#cbd5e1' }}>
                              {diag.generalization_gap !== undefined ? `${(Math.abs(diag.generalization_gap) * 100).toFixed(1)}%` : '-'}
                            </td>
                            {Object.entries(metrics)
                              .filter(([k]) => !['confusion_matrix', 'plot_data', 'fit_diagnosis', 'train_r2', 'train_rmse', 'train_mae', 'test_r2', 'test_rmse', 'train_accuracy', 'train_f1', 'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std', 'cv_r2_mean', 'cv_r2_std'].includes(k))
                              .map(([k, v]) => (
                                <td key={k} style={{ padding: '1rem' }}>{v}</td>
                              ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Confusion Matrix (Classification) */}
                {mlResults.metrics?.confusion_matrix && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ marginBottom: '1rem', color: 'white' }}>Confusion Matrix (Best Model: {mlResults.model_name})</h3>
                    <div style={{ display: 'inline-block', background: 'var(--bg-color)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {mlResults.metrics.confusion_matrix.map((row, i) => (
                          <div key={i} style={{ display: 'flex', gap: '6px' }}>
                            {row.map((cell, j) => {
                              const maxVal = Math.max(...mlResults.metrics.confusion_matrix.flat());
                              const intensity = Math.min(1, Math.max(0.15, cell / (maxVal || 1)));
                              const isDiagonal = i === j;
                              return (
                                <div key={j} style={{
                                  width: '56px', height: '56px',
                                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                                  background: isDiagonal ? `rgba(16, 185, 129, ${intensity})` : `rgba(239, 68, 68, ${intensity * 0.7})`,
                                  color: intensity > 0.5 ? 'white' : '#cbd5e1',
                                  fontWeight: 'bold', borderRadius: '6px', fontSize: '1rem'
                                }}>
                                  {cell}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                      <p style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        Green diagonal = correct predictions. Off-diagonal = misclassifications.
                      </p>
                    </div>
                  </div>
                )}

                {/* Actual vs Predicted Scatter (Regression) */}
                {mlResults.metrics?.plot_data && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ marginBottom: '1rem', color: 'white' }}>Actual vs Predicted Scatter Plot (Best Model)</h3>
                    <div style={{ height: '300px', width: '100%', background: 'var(--bg-color)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                          <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--text-secondary)" tick={{ fill: '#94a3b8' }} />
                          <YAxis type="number" dataKey="predicted" name="Predicted" stroke="var(--text-secondary)" tick={{ fill: '#94a3b8' }} />
                          <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--surface-color)', borderColor: 'var(--border-color)', color: 'white' }} />
                          <Scatter name="Predictions" data={mlResults.metrics.plot_data} fill="#818cf8" />
                          <ReferenceLine segment={[{ x: 'dataMin', y: 'dataMin' }, { x: 'dataMax', y: 'dataMax' }]} stroke="#10b981" strokeDasharray="3 3" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Batch Prediction Launchpad Container */}
                <div className="batch-predict-box">
                  <div>
                    <div style={{ fontWeight: 'bold', fontSize: '1.1rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>🔮</span> What is &quot;Batch Predict (Upload CSV)&quot;?
                    </div>
                    <p style={{ fontSize: '0.875rem', color: '#cbd5e1', marginTop: '0.35rem', maxWidth: '650px', lineHeight: '1.5' }}>
                      Batch Prediction takes an uploaded CSV file containing new, unlabelled customer or record rows and uses your trained <strong>{mlResults.model_name}</strong> to predict the target for every single row at once.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={handleDownloadTemplate} style={{ fontSize: '0.875rem' }}>
                      📥 Download CSV Template
                    </button>
                    <button
                      className="btn btn-primary"
                      onClick={() => document.getElementById('predictInput').click()}
                      disabled={loadingTask === 'predict'}
                      style={{ fontSize: '0.875rem' }}
                    >
                      {loadingTask === 'predict' ? 'Predicting...' : '📁 Upload CSV & Run Predictions'}
                    </button>
                  </div>
                </div>

              </div>
            )}

            {/* Model Explainability */}
            {explainResults && explainResults.explanations && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <h2 style={{ marginBottom: '1rem', color: 'var(--secondary)' }}>Model Explainability & Feature Drivers</h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontStyle: 'italic', fontSize: '0.875rem' }}>
                  {explainResults.explanations.notice}
                </p>
                {explainResults.explanations.feature_importances && (
                  <div>
                    <h3 style={{ marginBottom: '1rem' }}>Feature Importances (% Contribution to Prediction)</h3>
                    <div style={{ display: 'grid', gap: '0.75rem' }}>
                      {Object.entries(explainResults.explanations.feature_importances)
                        .sort((a, b) => b[1] - a[1])
                        .map(([feature, importance]) => (
                          <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '180px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', color: '#cbd5e1' }}>
                              {feature}
                            </div>
                            <div style={{ flex: 1, background: 'var(--bg-color)', height: '14px', borderRadius: '7px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                              <div style={{ width: `${Math.min(100, Math.max(0, importance * 100))}%`, height: '100%', background: 'linear-gradient(90deg, #6366f1, #ec4899)' }}></div>
                            </div>
                            <div style={{ width: '60px', textAlign: 'right', fontSize: '0.875rem', fontWeight: 'bold', color: 'white' }}>
                              {(importance * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                {explainResults.explanations.coefficients && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Linear Coefficients</h3>
                    <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.8', color: '#cbd5e1', fontSize: '0.9rem' }}>
                      {Object.entries(explainResults.explanations.coefficients).map(([feature, coef]) => (
                        <li key={feature}>
                          <strong>{feature}</strong>: <span style={{ color: coef > 0 ? '#34d399' : '#f87171' }}>{coef > 0 ? `+${coef.toFixed(4)}` : coef.toFixed(4)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Batch Predictions Results Table */}
            {predictionResults && predictionResults.predictions && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                  <div>
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>🔮</span> Batch Predictions Generated
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                      Showing predictions for {predictionResults.predictions.length} records. The newly calculated predictions are highlighted in green.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <input
                      type="text"
                      placeholder="Filter records..."
                      value={predictSearchQuery}
                      onChange={(e) => setPredictSearchQuery(e.target.value)}
                      style={{ background: 'var(--bg-color)', border: '1px solid var(--border-color)', color: 'white', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.875rem' }}
                    />
                    <button className="btn btn-primary" onClick={handleDownloadPredictions}>
                      📥 Download Predictions (CSV)
                    </button>
                  </div>
                </div>

                <div style={{ overflowX: 'auto', maxHeight: '420px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'var(--bg-color)' }}>
                    <thead style={{ position: 'sticky', top: 0, background: 'var(--surface-color)', zIndex: 10 }}>
                      <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <th style={{ padding: '0.75rem 1rem', color: '#34d399', fontWeight: 'bold' }}>⭐ PREDICTION</th>
                        {Object.keys(predictionResults.predictions[0])
                          .filter(col => col !== 'prediction' && !col.startsWith('probability_'))
                          .map(col => (
                            <th key={col} style={{ padding: '0.75rem 1rem', color: 'var(--text-secondary)' }}>{col}</th>
                          ))}
                        {Object.keys(predictionResults.predictions[0])
                          .filter(col => col.startsWith('probability_'))
                          .map(col => (
                            <th key={col} style={{ padding: '0.75rem 1rem', color: '#a5b4fc' }}>{col}</th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredPredictions.slice(0, 25).map((row, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                          <td style={{ padding: '0.75rem 1rem', fontWeight: 'bold', color: '#10b981', background: 'rgba(16, 185, 129, 0.08)' }}>
                            {String(row.prediction)}
                          </td>
                          {Object.keys(row)
                            .filter(col => col !== 'prediction' && !col.startsWith('probability_'))
                            .map((col, j) => (
                              <td key={j} style={{ padding: '0.75rem 1rem', fontSize: '0.875rem', color: '#cbd5e1' }}>
                                {String(row[col])}
                              </td>
                            ))}
                          {Object.keys(row)
                            .filter(col => col.startsWith('probability_'))
                            .map((col, j) => (
                              <td key={j} style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', color: '#a5b4fc' }}>
                                {typeof row[col] === 'number' ? `${(row[col] * 100).toFixed(1)}%` : String(row[col])}
                              </td>
                            ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {filteredPredictions.length > 25 && (
                  <p style={{ marginTop: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center', fontSize: '0.85rem' }}>
                    Showing first 25 of {filteredPredictions.length} records. Download CSV to inspect all predictions.
                  </p>
                )}
              </div>
            )}

            {/* Dataset Profile Grid */}
            <h2>Dataset Profile & Metadata</h2>
            <div className="stats-grid" style={{ marginBottom: '2rem' }}>
              <div className="stat-card">
                <div className="stat-value">{profile.dataset?.rows?.toLocaleString()}</div>
                <div className="stat-label">Total Rows</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{profile.dataset?.columns}</div>
                <div className="stat-label">Total Columns</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{profile.duplicates?.count || 0}</div>
                <div className="stat-label">Duplicate Rows</div>
              </div>
            </div>

            {/* Data Quality & Types */}
            <div className="glass-card" style={{ marginBottom: '2rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>Column Schema & Missing Values</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '1rem' }}>
                {profile.missing_values?.counts && Object.keys(profile.missing_values.counts).map(col => (
                  <div key={col} style={{ padding: '1rem', background: 'var(--surface-color)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: 'white' }}>{col}</div>
                    <div style={{ fontSize: '0.875rem', color: profile.missing_values.counts[col] > 0 ? 'var(--error)' : 'var(--success)' }}>
                      {profile.missing_values.counts[col] > 0 ? `⚠️ ${profile.missing_values.counts[col]} missing (${profile.missing_values.percentages[col]?.toFixed(1)}%)` : '✓ Complete (0 missing)'}
                    </div>
                    <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      Type: <code>{profile.data_types?.[col]}</code>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </main>

      {/* Floating Chat Agent Button & Panel */}
      {profile && (
        <>
          <button
            onClick={() => setIsChatOpen(!isChatOpen)}
            style={{
              position: 'fixed', bottom: '24px', right: '24px',
              width: '60px', height: '60px', borderRadius: '30px',
              background: 'linear-gradient(135deg, var(--primary), var(--secondary))',
              color: 'white', border: 'none', boxShadow: '0 8px 25px rgba(99, 102, 241, 0.4)',
              zIndex: 1001, cursor: 'pointer', fontSize: '24px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'transform 0.2s'
            }}
          >
            {isChatOpen ? '✕' : '💬'}
          </button>

          {isChatOpen && (
            <div className="chat-panel animate-fade-in" style={{
              position: 'fixed', bottom: '96px', right: '24px',
              width: '380px', maxHeight: '520px', background: 'var(--surface-color)',
              border: '1px solid var(--border-color)', borderRadius: '16px',
              display: 'flex', flexDirection: 'column', overflow: 'hidden',
              boxShadow: '0 20px 50px rgba(0,0,0,0.6)', zIndex: 1000
            }}>
              <div style={{ padding: '1rem 1.25rem', background: 'linear-gradient(135deg, var(--primary), var(--secondary))', color: 'white', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>✨ DataMind AI Agent</span>
                <span style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.2)', padding: '2px 8px', borderRadius: '10px' }}>Active</span>
              </div>
              <div style={{ height: '340px', overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
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
                    wordBreak: 'break-word',
                    color: 'white'
                  }}>
                    {msg.content}
                  </div>
                ))}
                {isChatting && <div style={{ fontSize: '0.85rem', color: '#a5b4fc', padding: '0.5rem', fontStyle: 'italic' }}>🤖 Agent is analyzing dataset...</div>}
              </div>
              <form onSubmit={handleSendMessage} style={{ display: 'flex', borderTop: '1px solid var(--border-color)', background: 'var(--bg-color)' }}>
                <input
                  type="text"
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  placeholder="Ask agent: e.g. 'clean outliers and train model'..."
                  style={{ flex: 1, background: 'transparent', border: 'none', padding: '1rem', color: 'white', outline: 'none', fontSize: '0.875rem' }}
                  disabled={isChatting}
                />
                <button type="submit" className="btn btn-primary" style={{ borderRadius: 0, padding: '0 1.25rem' }} disabled={isChatting}>
                  Send
                </button>
              </form>
            </div>
          )}
        </>
      )}

      {/* Interactive Documentation & Guide Modal */}
      {isDocModalOpen && (
        <div className="modal-overlay" onClick={() => setIsDocModalOpen(false)}>
          <div className="modal-content animate-fade-in" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <span>📖</span> DataMind AI User Guide & Documentation
              </div>
              <button className="modal-close-btn" onClick={() => setIsDocModalOpen(false)}>✕</button>
            </div>

            <div className="modal-tabs">
              <button
                className={`modal-tab ${activeDocTab === 'quickstart' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('quickstart')}
              >
                🚀 Quick Start & Workflow
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'eda' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('eda')}
              >
                📊 Understanding EDA & Charts
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'overfitting' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('overfitting')}
              >
                🧠 Overfitting vs Underfitting
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('batch')}
              >
                🔮 Batch Predict (Upload CSV)
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'api' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('api')}
              >
                ⚡ API Reference & Swagger
              </button>
            </div>

            <div className="modal-body">
              {activeDocTab === 'quickstart' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>End-to-End DataMind AI Workflow</h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    DataMind AI is an automated data science platform that takes your raw dataset and helps you profile, analyze, train, diagnose, and deploy machine learning models in seconds without writing code.
                  </p>

                  <div style={{ display: 'grid', gap: '1rem' }}>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>1. Upload Dataset (.csv or .xlsx)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Drag & drop your spreadsheet. DataMind profiles row counts, column data types, missing values, and duplicate records instantly.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>2. Run Plain-English EDA (Exploratory Data Analysis)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Produces clear, visual histograms showing value distribution ranges, highlights potential outliers, discovers high correlations, and generates plain-English takeaway insights.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>3. Auto-Train Machine Learning Models</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Select your target variable (e.g. Price, Churn, Species). DataMind automatically cleans data, encodes categories, scales numbers, trains competing algorithms, and picks the champion.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>4. Check Model Diagnostics (Overfitting & Underfitting)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Review the Train vs Test generalization score chart to verify your model isn’t just memorizing training data.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>5. Batch Predict on New Records</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Download the auto-generated CSV template, paste in new unseen records, and run predictions on thousands of rows at once.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'eda' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>How to Read EDA Charts & Numerical Distributions</h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    Exploratory Data Analysis helps you understand what the data looks like before training any algorithm.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.08)', padding: '1rem', borderRadius: '10px', border: '1px solid rgba(99, 102, 241, 0.25)' }}>
                      <strong style={{ color: '#818cf8' }}>📊 What is a Numerical Histogram?</strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        Unlike a simple average line, a histogram divides continuous values into 8–10 equal range &quot;bins&quot; along the horizontal X-axis and counts how many rows fall into each range along the vertical Y-axis. The taller the bar, the more records possess values in that bucket.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>📈 Distribution Shapes Explained</strong>
                      <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                        <li><strong>Normal / Bell-shaped:</strong> Most data clusters symmetrically around the center (mean ≈ median). Ideal for all standard machine learning models.</li>
                        <li><strong>Right-skewed (Tail on Highs):</strong> Most rows are small (e.g. typical salaries or house prices), but a small number of extremely large luxury values stretch out to the right.</li>
                        <li><strong>Left-skewed (Tail on Lows):</strong> Most rows cluster on higher values, with a few unusually small values stretching out to the left.</li>
                      </ul>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>⚠️ Outliers (IQR Method)</strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        An outlier is a data point located outside 1.5 times the Interquartile Range (IQR). It could represent a rare real-world occurrence (e.g., a billionaire in an income survey) or a data typo (e.g., Age = 999).
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>🔗 Feature Correlation</strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        Measured from -1.0 to +1.0. A correlation of +0.90 between two columns means that when one increases, the other almost always increases. Having redundant columns can confuse linear models.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'overfitting' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>Overfitting vs Underfitting: Plain-English Guide</h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    When machine learning models train, they must strike a balance between learning the patterns in the data and generalizing to new, unseen cases.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <h4 style={{ color: '#f87171', marginBottom: '0.5rem' }}>🔴 What is Overfitting? (High Variance)</h4>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                        <strong>Analogy:</strong> Imagine a student who memorizes every practice question and answer key word-for-word before an exam. On the practice test, they score 100%. But when given a real exam with new questions phrased slightly differently, they fail because they never learned the underlying concept.
                      </p>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        <strong>How to spot it in DataMind:</strong> The <em>Training Score</em> is very high (e.g. 99%), but the <em>Test Score</em> drops noticeably (e.g. 75%), leaving a large <strong>Generalization Gap (&gt; 12-15%)</strong>.
                      </p>
                      <div style={{ marginTop: '0.75rem', fontSize: '0.825rem', color: '#fca5a5' }}>
                        <strong>How to fix:</strong> Limit tree depth, add L1/L2 regularization, remove noisy features, or collect more training samples.
                      </div>
                    </div>

                    <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <h4 style={{ color: '#fbbf24', marginBottom: '0.5rem' }}>🟠 What is Underfitting? (High Bias)</h4>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                        <strong>Analogy:</strong> A student who barely opened the textbook and tries to guess answers using a single over-simplified rule. They score poorly on both the practice test and the real exam.
                      </p>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        <strong>How to spot it in DataMind:</strong> Both the <em>Training Score</em> and the <em>Test Score</em> are low (e.g. R² &lt; 0.30 or Accuracy &lt; 60%). The model is too simple to capture the complexity of the data.
                      </p>
                      <div style={{ marginTop: '0.75rem', fontSize: '0.825rem', color: '#fde68a' }}>
                        <strong>How to fix:</strong> Switch to non-linear algorithms (Random Forest or Extra Trees), engineer more predictive features, or reduce regularization.
                      </div>
                    </div>

                    <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <h4 style={{ color: '#34d399', marginBottom: '0.5rem' }}>🟢 What is a Good Fit? (Balanced)</h4>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                        Both Training and Test scores are high and closely matched (Generalization Gap &lt; 5-10%). The model learned generalized patterns that hold true on unseen data.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'batch' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>What Does &quot;Batch Predict (Upload CSV)&quot; Mean?</h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                    In machine learning, training a model is only the first step. The true business value comes when you use that trained model to make predictions on <strong>new records</strong> where the outcome is currently unknown.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <strong style={{ color: '#818cf8', fontSize: '1rem' }}>🔮 How Batch Prediction Works:</strong>
                      <ol style={{ paddingLeft: '1.25rem', marginTop: '0.75rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.8' }}>
                        <li>Train a model in DataMind on historical records (e.g. customer churn or house sales).</li>
                        <li>Click <strong>&quot;Download Batch Template&quot;</strong> to receive a clean CSV pre-populated with the exact feature column names your model requires.</li>
                        <li>Fill that CSV with your new rows of customer or product data (without the target column).</li>
                        <li>Click <strong>&quot;Batch Predict (Upload CSV)&quot;</strong> and upload that file.</li>
                        <li>The model runs predictions across all rows, adds a <code>prediction</code> column (and class probabilities), displays the table in the UI, and allows you to download the completed CSV!</li>
                      </ol>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong>Real-World Use Cases</strong>
                      <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                        <li><strong>Finance:</strong> Score 500 loan applicants to predict default probability.</li>
                        <li><strong>Healthcare:</strong> Screen incoming patient symptom surveys to predict risk level.</li>
                        <li><strong>Retail / E-Commerce:</strong> Predict sales demand for next quarter across 1,000 product SKUs.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'api' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>DataMind API Reference</h3>
                  <p style={{ marginBottom: '1rem', color: '#cbd5e1' }}>
                    DataMind exposes a high-performance REST API powered by FastAPI. You can interact with it programmatically or via Swagger UI:
                  </p>

                  <div style={{ marginBottom: '1.5rem' }}>
                    <a
                      href="http://localhost:8000/docs"
                      target="_blank"
                      rel="noreferrer"
                      className="btn btn-primary"
                      style={{ display: 'inline-flex', gap: '0.5rem', textDecoration: 'none' }}
                    >
                      <span>⚡</span> Open Interactive Swagger UI (http://localhost:8000/docs)
                    </a>
                  </div>

                  <div style={{ display: 'grid', gap: '0.75rem' }}>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>POST</span> <code>/datasets/upload</code> - Upload dataset and auto-profile
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>POST</span> <code>/datasets/&#123;id&#125;/eda</code> - Run automated plain-English EDA
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>POST</span> <code>/datasets/&#123;id&#125;/ml/train</code> - Train AutoML models with overfitting diagnostics
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#818cf8', fontWeight: 'bold' }}>GET</span> <code>/models/&#123;id&#125;/template</code> - Download blank CSV template for batch predict
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>POST</span> <code>/models/&#123;id&#125;/predict/batch</code> - Execute batch inference on uploaded CSV
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>POST</span> <code>/agent/chat</code> - Chat with LangGraph AI Data Science agent
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
