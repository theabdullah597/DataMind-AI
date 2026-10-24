'use client';
import { useState, useMemo, memo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import {
  Sparkles,
  UploadCloud,
  FileSpreadsheet,
  BarChart3,
  BarChart2,
  PieChart,
  Brain,
  HelpCircle,
  Activity,
  Trophy,
  CheckCircle2,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Bot,
  Send,
  Download,
  Search,
  BookOpen,
  ExternalLink,
  X,
  Layers,
  ChevronRight,
  Database,
  Table,
  Cpu,
  Lightbulb,
  Check,
  ShieldCheck,
  Compass
} from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ==========================================
// 1. ISOLATED CHAT AGENT (Prevents re-renders of the main page)
// ==========================================
const ChatAgentWidget = memo(function ChatAgentWidget({ datasetId }) {
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: "Hello! I'm your DataMind AI Agent. How can I help you explore or clean your dataset?" }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatting) return;

    const newMessages = [...chatMessages, { role: 'user', content: chatInput.trim() }];
    setChatMessages(newMessages);
    setChatInput('');
    setIsChatting(true);

    try {
      const response = await fetch(`${API_BASE}/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages, dataset_id: datasetId })
      });
      if (!response.ok) throw new Error('Chat agent request failed');
      const data = await response.json();
      setChatMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setIsChatOpen(!isChatOpen)}
        aria-label="Toggle DataMind AI Agent"
        style={{
          position: 'fixed', bottom: '24px', right: '24px',
          width: '56px', height: '56px', borderRadius: '28px',
          background: 'linear-gradient(135deg, var(--primary), var(--secondary))',
          color: 'white', border: 'none', boxShadow: '0 8px 25px rgba(99, 102, 241, 0.45)',
          zIndex: 1001, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
        }}
      >
        {isChatOpen ? <X size={24} /> : <Bot size={24} />}
      </button>

      {isChatOpen && (
        <div className="chat-panel animate-fade-in" style={{
          position: 'fixed', bottom: '92px', right: '24px',
          width: '380px', maxHeight: '520px', background: 'var(--surface-color)',
          border: '1px solid var(--border-color)', borderRadius: '16px',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
          boxShadow: '0 20px 50px rgba(0,0,0,0.6)', zIndex: 1000
        }}>
          <div style={{ padding: '0.875rem 1.25rem', background: 'linear-gradient(135deg, var(--primary), var(--secondary))', color: 'white', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.95rem' }}>
              <Bot size={18} /> DataMind AI Agent
            </span>
            <span style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.2)', padding: '2px 8px', borderRadius: '10px' }}>Active</span>
          </div>

          <div style={{ height: '340px', overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }} className="custom-scrollbar">
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
                fontSize: '0.85rem',
                lineHeight: '1.5',
                wordBreak: 'break-word',
                color: 'white'
              }}>
                {msg.content}
              </div>
            ))}
            {isChatting && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#a5b4fc', padding: '0.5rem', fontStyle: 'italic' }}>
                <Sparkles size={14} /> Agent is analyzing dataset...
              </div>
            )}
          </div>

          <form onSubmit={handleSendMessage} style={{ display: 'flex', borderTop: '1px solid var(--border-color)', background: 'var(--bg-color)' }}>
            <input
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder="Ask agent: e.g. 'clean outliers and train model'..."
              style={{ flex: 1, background: 'transparent', border: 'none', padding: '0.875rem 1rem', color: 'white', outline: 'none', fontSize: '0.85rem' }}
              disabled={isChatting}
            />
            <button type="submit" className="btn btn-primary" style={{ borderRadius: 0, padding: '0 1.25rem' }} disabled={isChatting || !chatInput.trim()}>
              <Send size={15} />
            </button>
          </form>
        </div>
      )}
    </>
  );
});

// ==========================================
// 2. SAFE CONFUSION MATRIX (Fixes RangeError: Maximum call stack size exceeded)
// ==========================================
const SafeConfusionMatrix = memo(function SafeConfusionMatrix({ matrix, modelName }) {
  if (!matrix || !Array.isArray(matrix) || matrix.length === 0) return null;

  // Safe iterative calculation of max value (O(N) without spreading large arrays onto call stack)
  let maxVal = 1;
  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i];
    if (Array.isArray(row)) {
      for (let j = 0; j < row.length; j++) {
        const val = Number(row[j]);
        if (!isNaN(val) && val > maxVal) {
          maxVal = val;
        }
      }
    }
  }

  const numClasses = matrix.length;
  const isLarge = numClasses > 10;
  const displayMatrix = isLarge ? matrix.slice(0, 10).map(r => (Array.isArray(r) ? r.slice(0, 10) : [])) : matrix;

  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 style={{ margin: 0, color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.15rem' }}>
          <Table size={18} color="var(--primary)" /> Confusion Matrix (Best Model: {modelName})
        </h3>
        {isLarge && (
          <span className="badge badge-info" style={{ fontSize: '0.75rem' }}>
            Displaying top 10 of {numClasses} classes (Scrollable)
          </span>
        )}
      </div>

      <div style={{
        background: 'var(--bg-color)',
        padding: '1.25rem',
        borderRadius: '12px',
        border: '1px solid var(--border-color)',
        maxWidth: '100%',
        overflowX: 'auto'
      }} className="custom-scrollbar">
        <div style={{ display: 'inline-flex', flexDirection: 'column', gap: '6px', minWidth: 'min-content' }}>
          {displayMatrix.map((row, i) => (
            <div key={i} style={{ display: 'flex', gap: '6px' }}>
              {Array.isArray(row) && row.map((cell, j) => {
                const numericCell = Number(cell) || 0;
                const intensity = Math.min(1, Math.max(0.15, numericCell / (maxVal || 1)));
                const isDiagonal = i === j;
                return (
                  <div
                    key={j}
                    title={`True class ${i}, Predicted class ${j}: ${numericCell} cases`}
                    style={{
                      width: numClasses > 6 ? '46px' : '56px',
                      height: numClasses > 6 ? '46px' : '56px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: isDiagonal
                        ? `rgba(16, 185, 129, ${intensity})`
                        : numericCell === 0
                        ? 'rgba(255, 255, 255, 0.03)'
                        : `rgba(239, 68, 68, ${intensity * 0.7})`,
                      color: intensity > 0.5 ? 'white' : '#cbd5e1',
                      fontWeight: 'bold',
                      borderRadius: '6px',
                      fontSize: numClasses > 6 ? '0.85rem' : '1rem',
                      border: isDiagonal ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(255, 255, 255, 0.05)'
                    }}
                  >
                    {numericCell}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <p style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <span style={{ color: '#34d399', fontWeight: '600' }}>Green diagonal</span> = correct predictions.
          <span style={{ color: '#f87171', fontWeight: '600', marginLeft: '0.5rem' }}>Red off-diagonal</span> = misclassifications.
        </p>
      </div>
    </div>
  );
});

// ==========================================
// 3. MAIN COMPONENT
// ==========================================
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

  // Performance Controls for Charts (Prevents browser freeze on datasets with 20+ columns)
  const [numSearch, setNumSearch] = useState('');
  const [showAllNum, setShowAllNum] = useState(false);
  const [catSearch, setCatSearch] = useState('');
  const [showAllCat, setShowAllCat] = useState(false);

  // Documentation Modal State
  const [isDocModalOpen, setIsDocModalOpen] = useState(false);
  const [activeDocTab, setActiveDocTab] = useState('quickstart');

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
      const response = await fetch(`${API_BASE}/datasets/upload`, {
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
      // Intelligently pick the best target column
      if (data.profile?.dataset?.column_names?.length) {
        const cols = data.profile.dataset.column_names;
        const uVals = data.profile.unique_values || {};
        const dTypes = data.profile.data_types || {};

        let bestCol = '';
        // 1. Check for common target keywords
        const kwList = ['target', 'label', 'class', 'churn', 'status', 'outcome', 'category', 'survived', 'price', 'default'];
        for (const kw of kwList) {
          const match = cols.find(c => c.toLowerCase().includes(kw));
          if (match && (uVals[match] === undefined || uVals[match] <= 50 || dTypes[match] !== 'object')) {
            bestCol = match;
            break;
          }
        }
        // 2. Check for low-cardinality discrete category (2 to 20 classes)
        if (!bestCol) {
          const discrete = cols.find(c => uVals[c] !== undefined && uVals[c] >= 2 && uVals[c] <= 20);
          if (discrete) bestCol = discrete;
        }
        // 3. Check for numeric column
        if (!bestCol) {
          const numCol = cols.find(c => dTypes[c] && ['int', 'float', 'number'].some(t => String(dTypes[c]).includes(t)));
          if (numCol) bestCol = numCol;
        }
        setSelectedTarget(bestCol || cols[0]);
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
      const response = await fetch(`${API_BASE}/datasets/${datasetId}/eda${targetQuery}`, {
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
    const target = selectedTarget || profile?.dataset?.column_names?.[profile.dataset.column_names.length - 1];
    setLoadingTask('ml');
    setError('');
    try {
      const response = await fetch(`${API_BASE}/datasets/${datasetId}/ml/train`, {
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
      const response = await fetch(`${API_BASE}/models/${mlResults.model_id}/explain`);
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
      const response = await fetch(`${API_BASE}/models/${mlResults.model_id}/predict/batch`, {
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
      e.target.value = '';
    }
  };

  const handleDownloadTemplate = () => {
    if (!mlResults || !mlResults.model_id) return;
    window.open(`${API_BASE}/models/${mlResults.model_id}/template`, '_blank');
  };

  const handleDownloadDataset = () => {
    window.open(`${API_BASE}/datasets/${datasetId}/download`, '_blank');
  };

  const handleDownloadModel = () => {
    window.open(`${API_BASE}/models/${mlResults.model_id}/download`, '_blank');
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
    setNumSearch('');
    setCatSearch('');
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

  // Numerical entries for EDA (Filtered & Paginated for High Performance)
  const numericalEntries = useMemo(() => {
    if (!edaResults?.numerical_analysis) return [];
    const all = Object.entries(edaResults.numerical_analysis);
    if (!numSearch.trim()) {
      return showAllNum ? all : all.slice(0, 6);
    }
    const q = numSearch.toLowerCase();
    return all.filter(([col]) => col.toLowerCase().includes(q));
  }, [edaResults, numSearch, showAllNum]);

  const totalNumColumns = useMemo(() => {
    return edaResults?.numerical_analysis ? Object.keys(edaResults.numerical_analysis).length : 0;
  }, [edaResults]);

  // Categorical entries for EDA (Filtered & Paginated for High Performance)
  const categoricalEntries = useMemo(() => {
    if (!edaResults?.categorical_analysis) return [];
    const all = Object.entries(edaResults.categorical_analysis);
    if (!catSearch.trim()) {
      return showAllCat ? all : all.slice(0, 6);
    }
    const q = catSearch.toLowerCase();
    return all.filter(([col]) => col.toLowerCase().includes(q));
  }, [edaResults, catSearch, showAllCat]);

  const totalCatColumns = useMemo(() => {
    return edaResults?.categorical_analysis ? Object.keys(edaResults.categorical_analysis).length : 0;
  }, [edaResults]);

  return (
    <div className="container" style={{ paddingBottom: '100px' }}>
      {/* Top Bar Header */}
      <header className="header animate-fade-in">
        <div className="logo" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }} onClick={handleReset}>
          <Sparkles size={24} color="var(--primary)" />
          <span>DataMind AI</span>
          <span style={{ fontSize: '0.75rem', background: 'var(--primary)', padding: '2px 8px', borderRadius: '6px', color: 'white', verticalAlign: 'middle', fontWeight: 'bold' }}>PRO</span>
        </div>
        <nav style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            className="btn btn-secondary"
            onClick={() => setIsDocModalOpen(true)}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}
          >
            <BookOpen size={16} color="var(--primary)" /> Documentation & Guide
          </button>
          <a
            href={`${API_BASE}/docs`}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none', fontSize: '0.9rem' }}
          >
            <Cpu size={16} color="var(--secondary)" /> API Swagger
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
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                <UploadCloud size={24} color="var(--primary)" /> Upload Dataset
              </h2>
              <div
                className="file-upload"
                onClick={() => document.getElementById('fileInput').click()}
              >
                <div className="file-upload-icon" style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                  <FileSpreadsheet size={48} color="var(--primary)" />
                </div>
                <h3>{file ? file.name : 'Click to select CSV or Excel file'}</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.875rem' }}>
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
                style={{ width: '100%', margin: '1rem 0', gap: '0.5rem' }}
                onClick={handleUpload}
                disabled={uploading}
              >
                {uploading ? (
                  <>
                    <Sparkles size={18} /> Analyzing Dataset...
                  </>
                ) : (
                  <>
                    <Activity size={18} /> Analyze Dataset Now
                  </>
                )}
              </button>
            </div>

            <div className="features-grid">
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <BarChart3 size={20} color="var(--primary)" /> Understandable EDA
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Clear, jargon-free distribution histograms and automated plain-English data insights that anyone can interpret.
                </p>
              </div>
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Brain size={20} color="var(--secondary)" /> Overfitting Detection
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Automatic Train vs Test performance comparisons so you immediately know if your model overfits, underfits, or generalizes well.
                </p>
              </div>
              <div className="glass-card">
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layers size={20} color="#34d399" /> Batch Predictions
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
              <button className="btn btn-secondary" onClick={handleReset} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}>
                <UploadCloud size={15} /> Upload New Dataset
              </button>

              {/* Target column selector */}
              {profile.dataset?.column_names && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Target Column:</label>
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    style={{ background: 'var(--bg-color)', color: 'white', border: '1px solid var(--border-color)', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.9rem', outline: 'none' }}
                  >
                    {profile.dataset.column_names.map(c => {
                      const uCount = profile.unique_values?.[c];
                      const dType = profile.data_types?.[c];
                      const isTextCol = uCount > 50 && dType === 'object';
                      return (
                        <option key={c} value={c}>
                          {c} {uCount !== undefined ? `(${uCount} unique${isTextCol ? ' - Text/ID' : ''})` : ''}
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              <button className="btn btn-primary" onClick={handleRunEDA} disabled={loadingTask === 'eda'} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}>
                <BarChart3 size={15} />
                {loadingTask === 'eda' ? 'Generating EDA...' : 'Run Plain-English EDA'}
              </button>

              <button className="btn btn-primary" onClick={handleTrainModel} disabled={loadingTask === 'ml'} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}>
                <Brain size={15} />
                {loadingTask === 'ml' ? 'Training AutoML...' : `Train Models on "${selectedTarget || 'Target'}"`}
              </button>

              {mlResults && (
                <>
                  <button className="btn btn-secondary" onClick={handleExplainModel} disabled={loadingTask === 'explain'} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}>
                    <Lightbulb size={15} color="#fbbf24" />
                    {loadingTask === 'explain' ? 'Explaining...' : 'Explain Features'}
                  </button>
                  <button className="btn btn-secondary" onClick={handleDownloadTemplate} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}>
                    <Download size={15} /> Download Batch Template
                  </button>
                  <div style={{ position: 'relative' }}>
                    <input type="file" id="predictInput" style={{ display: 'none' }} accept=".csv, .xlsx" onChange={handlePredictBatch} />
                    <button
                      className="btn btn-secondary"
                      onClick={() => document.getElementById('predictInput').click()}
                      disabled={loadingTask === 'predict'}
                      style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'var(--primary)', color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem' }}
                    >
                      <Layers size={15} />
                      {loadingTask === 'predict' ? 'Predicting Rows...' : 'Batch Predict (Upload CSV)'}
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* Warning if user selects a high-cardinality text/ID column as target */}
            {profile.unique_values?.[selectedTarget] > 50 && profile.data_types?.[selectedTarget] === 'object' && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.35)',
                color: '#fca5a5',
                padding: '0.85rem 1.25rem',
                borderRadius: '10px',
                marginBottom: '1.75rem',
                fontSize: '0.875rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                lineHeight: '1.5'
              }}>
                <AlertTriangle size={20} color="#f87171" style={{ flexShrink: 0 }} />
                <div>
                  <strong>Invalid Target Column Warning:</strong> &quot;{selectedTarget}&quot; has {profile.unique_values[selectedTarget]} unique text descriptions/strings.
                  Classification algorithms require discrete categories with multiple examples per class (e.g. <code>Class Index</code> with 4 classes).
                  Training on raw text sentences results in 0% accuracy. Please select a category column like <strong>Class Index</strong>.
                </div>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {/* EDA Results Section */}
            {edaResults && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                  <div>
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.4rem' }}>
                      <BarChart3 size={24} color="var(--primary)" /> Plain-English EDA Insights
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
                          fontWeight: '600',
                          transition: 'all 0.15s ease'
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
                        <div style={{ background: 'rgba(16, 185, 129, 0.08)', borderLeft: '3px solid var(--success)', padding: '0.5rem 0.75rem', borderRadius: '0 6px 6px 0', fontSize: '0.8rem', color: '#a7f3d0', display: 'flex', alignItems: 'flex-start', gap: '0.4rem' }}>
                          <Lightbulb size={14} style={{ marginTop: '2px', flexShrink: 0 }} />
                          <div><strong>Action:</strong> {ins.recommendation}</div>
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
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                      <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <BarChart2 size={20} color="var(--primary)" /> Numerical Distributions & Histograms
                      </h3>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                        Value frequencies and distribution shapes. Showing {numericalEntries.length} of {totalNumColumns} columns.
                      </p>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <div style={{ position: 'relative' }}>
                        <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                        <input
                          type="text"
                          placeholder="Search column..."
                          value={numSearch}
                          onChange={(e) => setNumSearch(e.target.value)}
                          style={{
                            background: 'var(--bg-color)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '6px',
                            color: 'white',
                            padding: '0.4rem 0.6rem 0.4rem 2rem',
                            fontSize: '0.825rem',
                            outline: 'none'
                          }}
                        />
                      </div>
                      {totalNumColumns > 6 && !numSearch && (
                        <button
                          className="btn btn-secondary"
                          onClick={() => setShowAllNum(!showAllNum)}
                          style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
                        >
                          {showAllNum ? 'Show Less (6)' : `Show All (${totalNumColumns})`}
                        </button>
                      )}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1.5rem' }}>
                    {numericalEntries.map(([col, stats]) => (
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

                        {/* Fast Recharts Histogram (isAnimationActive false for snappy response) */}
                        <div style={{ height: '200px', width: '100%', marginTop: '0.5rem' }}>
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
                                tick={{ fontSize: 10, fill: '#94a3b8' }}
                                allowDecimals={false}
                              />
                              <RechartsTooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                      <div style={{ background: '#1e293b', border: '1px solid #475569', padding: '6px 10px', borderRadius: '6px', color: '#f8fafc', fontSize: '0.8rem' }}>
                                        <div style={{ fontWeight: 'bold', color: '#818cf8', marginBottom: '2px' }}>Range: {data.bin}</div>
                                        <div>Count: <span style={{ fontWeight: 'bold', color: 'white' }}>{data.count}</span> rows</div>
                                        {data.percentage !== undefined && <div>Share: {data.percentage}% of data</div>}
                                      </div>
                                    );
                                  }
                                  return null;
                                }}
                              />
                              <Bar dataKey="count" fill="url(#histGradient)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
                          <strong>Takeaway:</strong> {stats.simple_summary || `${col} ranges from ${stats.min} to ${stats.max} with an average of ${stats.mean}.`}
                          {stats.shape_desc && <div style={{ marginTop: '2px', opacity: 0.9 }}>{stats.shape_desc}</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Categorical Distributions */}
                {categoricalEntries.length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '2rem', marginBottom: '2.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <PieChart size={20} color="var(--secondary)" /> Categorical Distributions
                        </h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                          Breakdown of unique categories, counts, and relative percentages.
                        </p>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{ position: 'relative' }}>
                          <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                          <input
                            type="text"
                            placeholder="Search category..."
                            value={catSearch}
                            onChange={(e) => setCatSearch(e.target.value)}
                            style={{
                              background: 'var(--bg-color)',
                              border: '1px solid var(--border-color)',
                              borderRadius: '6px',
                              color: 'white',
                              padding: '0.4rem 0.6rem 0.4rem 2rem',
                              fontSize: '0.825rem',
                              outline: 'none'
                            }}
                          />
                        </div>
                        {totalCatColumns > 6 && !catSearch && (
                          <button
                            className="btn btn-secondary"
                            onClick={() => setShowAllCat(!showAllCat)}
                            style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
                          >
                            {showAllCat ? 'Show Less (6)' : `Show All (${totalCatColumns})`}
                          </button>
                        )}
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1.5rem' }}>
                      {categoricalEntries.map(([col, data]) => {
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

                            <div style={{ height: '200px', width: '100%', marginTop: '0.5rem' }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 25, left: -10 }}>
                                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                                  <XAxis
                                    dataKey="name"
                                    stroke="var(--text-secondary)"
                                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                                    angle={-25}
                                    textAnchor="end"
                                  />
                                  <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 10, fill: '#94a3b8' }} allowDecimals={false} />
                                  <RechartsTooltip
                                    content={({ active, payload }) => {
                                      if (active && payload && payload.length) {
                                        const d = payload[0].payload;
                                        return (
                                          <div style={{ background: '#1e293b', border: '1px solid #475569', padding: '6px 10px', borderRadius: '6px', color: '#f8fafc', fontSize: '0.8rem' }}>
                                            <div style={{ fontWeight: 'bold', color: '#ec4899', marginBottom: '2px' }}>Category: {d.name}</div>
                                            <div>Count: <span style={{ fontWeight: 'bold' }}>{d.value}</span> rows</div>
                                            <div>Percentage: <span style={{ fontWeight: 'bold' }}>{d.pct}%</span></div>
                                          </div>
                                        );
                                      }
                                      return null;
                                    }}
                                  />
                                  <Bar dataKey="value" fill="url(#catGradient)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
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
                      <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Activity size={20} color="#818cf8" /> Strong Feature Relationships (Correlations)
                      </h3>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                        Columns that move together. High correlation (&gt; 0.70) indicates shared predictive information.
                      </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
                      {edaResults.correlation_analysis.strong_correlations.map((corr, idx) => (
                        <div key={idx} style={{ background: 'var(--bg-color)', border: '1px solid var(--border-color)', padding: '1rem', borderRadius: '10px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: '600', color: 'white', fontSize: '0.9rem' }}>{corr.column_1} ↔ {corr.column_2}</span>
                            <span className={`badge ${corr.correlation > 0 ? 'badge-tip' : 'badge-alert'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                              {corr.correlation > 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                              {corr.correlation}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
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
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.4rem' }}>
                      <Brain size={24} color="var(--secondary)" /> AutoML Results & Training Diagnostics
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
                      Models trained and evaluated against unseen test records with automated overfitting detection.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={handleDownloadDataset} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
                      <Download size={14} /> Download Cleaned Data
                    </button>
                    <button className="btn btn-primary" onClick={handleDownloadModel} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
                      <Download size={14} /> Download Best Model (.joblib)
                    </button>
                  </div>
                </div>

                {/* Best Model Summary Card */}
                <div className="stat-card" style={{ marginBottom: '2rem', background: 'rgba(30, 41, 59, 0.8)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                      <div className="stat-value" style={{ fontSize: '1.5rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Trophy size={24} color="#fbbf24" /> Best Model: <span style={{ color: 'var(--primary)' }}>{mlResults.model_name}</span>
                      </div>
                      <div className="stat-label" style={{ fontSize: '0.95rem', marginTop: '0.5rem' }}>
                        Task: <strong style={{ textTransform: 'capitalize', color: 'white' }}>{mlResults.task_type}</strong> | Target: <strong style={{ color: 'var(--secondary)' }}>{mlResults.target_column || selectedTarget}</strong>
                      </div>
                      <div style={{ marginTop: '0.75rem', color: '#34d399', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <CheckCircle2 size={16} /> {mlResults.recommendation_reason}
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
                    <Brain size={18} color="var(--primary)" /> How to Identify if a Model Overfits or Underfits
                  </h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem', lineHeight: '1.6' }}>
                    A good machine learning model must perform well on <strong>unseen test data</strong>, not just the training data it memorized. Check the Train vs Test scores below:
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#34d399', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <CheckCircle2 size={16} /> Good Fit (Balanced)
                      </div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1', lineHeight: '1.5' }}>
                        Train and Test scores are both strong and close together (Gap &lt; 10%). The model learned true patterns and generalizes reliably.
                      </div>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#f87171', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <AlertTriangle size={16} /> Overfitting (High Variance)
                      </div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1', lineHeight: '1.5' }}>
                        Train score is high (e.g. 98%), but Test score is noticeably lower. The model memorized training noise like cramming for an exam.
                      </div>
                    </div>
                    <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '8px', padding: '1rem' }}>
                      <div style={{ fontWeight: 'bold', color: '#fbbf24', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <HelpCircle size={16} /> Underfitting (High Bias)
                      </div>
                      <div style={{ fontSize: '0.825rem', color: '#cbd5e1', lineHeight: '1.5' }}>
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
                        <div style={{ fontWeight: 'bold', color: 'white', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <BarChart3 size={16} color="var(--primary)" /> Train Score vs Test Score Comparison (Generalization Gap)
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

                    <div style={{ height: '240px', width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={modelFitChartData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                          <XAxis dataKey="shortName" stroke="var(--text-secondary)" tick={{ fontSize: 11, fill: '#cbd5e1' }} />
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
                          <Bar dataKey="trainScore" name="Training Score" fill="#818cf8" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                          <Bar dataKey="testScore" name="Test Score" fill="#10b981" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Model Comparison Table */}
                <h3 style={{ marginBottom: '1rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.15rem' }}>
                  <Table size={18} color="var(--primary)" /> Detailed Model Comparison Table
                </h3>
                <div style={{ overflowX: 'auto', marginBottom: '2rem' }} className="custom-scrollbar">
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'var(--bg-color)', borderRadius: '8px', overflow: 'hidden' }}>
                    <thead>
                      <tr style={{ background: 'var(--surface-color)', borderBottom: '1px solid var(--border-color)' }}>
                        <th style={{ padding: '0.85rem 1rem' }}>Model</th>
                        <th style={{ padding: '0.85rem 1rem' }}>Fit Diagnosis</th>
                        <th style={{ padding: '0.85rem 1rem' }}>Train Score</th>
                        <th style={{ padding: '0.85rem 1rem' }}>Test Score</th>
                        <th style={{ padding: '0.85rem 1rem' }}>Gap</th>
                        {Object.keys(Object.values(mlResults.all_results)[0])
                          .filter(k => !['confusion_matrix', 'plot_data', 'fit_diagnosis', 'train_r2', 'train_rmse', 'train_mae', 'test_r2', 'test_rmse', 'train_accuracy', 'train_f1', 'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std', 'cv_r2_mean', 'cv_r2_std'].includes(k))
                          .map(k => (
                            <th key={k} style={{ padding: '0.85rem 1rem', textTransform: 'capitalize' }}>{k}</th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(mlResults.all_results).map(([modelName, metrics]) => {
                        const diag = metrics.fit_diagnosis || {};
                        const isBest = modelName === mlResults.model_name;
                        return (
                          <tr key={modelName} style={{ borderBottom: '1px solid var(--border-color)', background: isBest ? 'rgba(99, 102, 241, 0.08)' : 'transparent' }}>
                            <td style={{ padding: '0.85rem 1rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                              {modelName} {isBest && <Trophy size={16} color="#fbbf24" />}
                            </td>
                            <td style={{ padding: '0.85rem 1rem' }}>
                              <span className={`badge ${
                                diag.status === 'Good Fit' ? 'badge-good-fit' :
                                diag.status === 'Overfitting' ? 'badge-overfitting' :
                                diag.status === 'Underfitting' ? 'badge-underfitting' : 'badge-info'
                              }`}>
                                {diag.badge || diag.status || 'Evaluated'}
                              </span>
                            </td>
                            <td style={{ padding: '0.85rem 1rem', color: '#818cf8', fontWeight: '600' }}>
                              {metrics.train_accuracy !== undefined ? `${(metrics.train_accuracy * 100).toFixed(1)}%` : metrics.train_r2 !== undefined ? metrics.train_r2 : '-'}
                            </td>
                            <td style={{ padding: '0.85rem 1rem', color: '#34d399', fontWeight: '600' }}>
                              {metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}%` : metrics.r2_score !== undefined ? metrics.r2_score : metrics.rmse !== undefined ? metrics.rmse : '-'}
                            </td>
                            <td style={{ padding: '0.85rem 1rem', color: diag.generalization_gap > 0.12 ? '#f87171' : '#cbd5e1' }}>
                              {diag.generalization_gap !== undefined ? `${(Math.abs(diag.generalization_gap) * 100).toFixed(1)}%` : '-'}
                            </td>
                            {Object.entries(metrics)
                              .filter(([k]) => !['confusion_matrix', 'plot_data', 'fit_diagnosis', 'train_r2', 'train_rmse', 'train_mae', 'test_r2', 'test_rmse', 'train_accuracy', 'train_f1', 'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std', 'cv_r2_mean', 'cv_r2_std'].includes(k))
                              .map(([k, v]) => (
                                <td key={k} style={{ padding: '0.85rem 1rem' }}>
                                  {typeof v === 'number' ? v.toFixed(4) : String(v)}
                                </td>
                              ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Safe Confusion Matrix (Prevents Stack Overflow) */}
                {mlResults.metrics?.confusion_matrix && (
                  <SafeConfusionMatrix
                    matrix={mlResults.metrics.confusion_matrix}
                    modelName={mlResults.model_name}
                  />
                )}

                {/* Actual vs Predicted Plot (Regression) */}
                {mlResults.metrics?.plot_data && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ marginBottom: '1rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.15rem' }}>
                      <Activity size={18} color="var(--primary)" /> Actual vs Predicted Residual Plot
                    </h3>
                    <div style={{ height: '280px', width: '100%', background: 'var(--bg-color)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                          <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--text-secondary)" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                          <YAxis type="number" dataKey="predicted" name="Predicted" stroke="var(--text-secondary)" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                          <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter name="Predictions" data={mlResults.metrics.plot_data} fill="#818cf8" isAnimationActive={false} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Batch Prediction Launchpad Container */}
                <div className="batch-predict-box">
                  <div>
                    <div style={{ fontWeight: 'bold', fontSize: '1.1rem', color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Layers size={20} color="var(--primary)" /> What is &quot;Batch Predict (Upload CSV)&quot;?
                    </div>
                    <p style={{ fontSize: '0.875rem', color: '#cbd5e1', marginTop: '0.35rem', maxWidth: '650px', lineHeight: '1.5' }}>
                      Batch Prediction takes an uploaded CSV file containing new, unlabelled customer or record rows and uses your trained <strong>{mlResults.model_name}</strong> to predict the target for every single row at once.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={handleDownloadTemplate} style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Download size={14} /> Download CSV Template
                    </button>
                    <button
                      className="btn btn-primary"
                      onClick={() => document.getElementById('predictInput').click()}
                      disabled={loadingTask === 'predict'}
                      style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
                    >
                      <Layers size={14} />
                      {loadingTask === 'predict' ? 'Predicting...' : 'Upload CSV & Run Predictions'}
                    </button>
                  </div>
                </div>

              </div>
            )}

            {/* Model Explainability */}
            {explainResults && explainResults.explanations && (
              <div className="glass-card animate-fade-in" style={{ marginBottom: '2.5rem' }}>
                <h2 style={{ marginBottom: '1rem', color: 'var(--secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.3rem' }}>
                  <Lightbulb size={22} color="#fbbf24" /> Model Explainability & Feature Drivers
                </h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontStyle: 'italic', fontSize: '0.875rem' }}>
                  {explainResults.explanations.notice}
                </p>
                {explainResults.explanations.feature_importances && (
                  <div>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1.05rem', color: 'white' }}>Feature Importances (% Contribution to Prediction)</h3>
                    <div style={{ display: 'grid', gap: '0.75rem' }}>
                      {Object.entries(explainResults.explanations.feature_importances)
                        .sort((a, b) => b[1] - a[1])
                        .map(([feature, importance]) => (
                          <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '180px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.875rem', color: '#cbd5e1' }}>
                              {feature}
                            </div>
                            <div style={{ flex: 1, background: 'var(--bg-color)', height: '12px', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                              <div style={{ width: `${Math.min(100, Math.max(0, importance * 100))}%`, height: '100%', background: 'linear-gradient(90deg, #6366f1, #ec4899)' }}></div>
                            </div>
                            <div style={{ width: '60px', textAlign: 'right', fontSize: '0.85rem', fontWeight: 'bold', color: 'white' }}>
                              {(importance * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                {explainResults.explanations.coefficients && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1.05rem', color: 'white' }}>Linear Coefficients</h3>
                    <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.8', color: '#cbd5e1', fontSize: '0.875rem' }}>
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
                    <h2 style={{ color: 'var(--secondary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.3rem' }}>
                      <Layers size={22} color="var(--primary)" /> Batch Predictions Generated
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                      Showing predictions for {predictionResults.predictions.length} records. The newly calculated predictions are highlighted in green.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div style={{ position: 'relative' }}>
                      <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                      <input
                        type="text"
                        placeholder="Filter records..."
                        value={predictSearchQuery}
                        onChange={(e) => setPredictSearchQuery(e.target.value)}
                        style={{ background: 'var(--bg-color)', border: '1px solid var(--border-color)', color: 'white', padding: '0.45rem 0.75rem 0.45rem 2rem', borderRadius: '6px', fontSize: '0.85rem', outline: 'none' }}
                      />
                    </div>
                    <button className="btn btn-primary" onClick={handleDownloadPredictions} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
                      <Download size={14} /> Download Predictions (CSV)
                    </button>
                  </div>
                </div>

                <div style={{ overflowX: 'auto', maxHeight: '420px', borderRadius: '8px', border: '1px solid var(--border-color)' }} className="custom-scrollbar">
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'var(--bg-color)' }}>
                    <thead style={{ position: 'sticky', top: 0, background: 'var(--surface-color)', zIndex: 10 }}>
                      <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <th style={{ padding: '0.75rem 1rem', color: '#34d399', fontWeight: 'bold' }}>PREDICTION</th>
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
                              <td key={j} style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', color: '#cbd5e1' }}>
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
            <h2 style={{ fontSize: '1.3rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Database size={20} color="var(--primary)" /> Dataset Profile & Metadata
            </h2>
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
              <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.15rem' }}>
                <Table size={18} color="var(--primary)" /> Column Schema & Missing Values
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '1rem' }}>
                {profile.missing_values?.counts && Object.keys(profile.missing_values.counts).map(col => (
                  <div key={col} style={{ padding: '1rem', background: 'var(--surface-color)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: 'white', fontSize: '0.9rem' }}>{col}</div>
                    <div style={{ fontSize: '0.85rem', color: profile.missing_values.counts[col] > 0 ? 'var(--error)' : 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      {profile.missing_values.counts[col] > 0 ? (
                        <>
                          <AlertTriangle size={14} />
                          {profile.missing_values.counts[col]} missing ({profile.missing_values.percentages[col]?.toFixed(1)}%)
                        </>
                      ) : (
                        <>
                          <CheckCircle2 size={14} /> Complete (0 missing)
                        </>
                      )}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      Type: <code style={{ color: '#cbd5e1' }}>{profile.data_types?.[col]}</code>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </main>

      {/* Standalone Chat Agent Widget (Prevents parent re-render lag) */}
      {profile && <ChatAgentWidget datasetId={datasetId} />}

      {/* Interactive Responsive Documentation & Guide Modal */}
      {isDocModalOpen && (
        <div className="modal-overlay" onClick={() => setIsDocModalOpen(false)}>
          <div className="modal-content animate-fade-in" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <BookOpen size={20} color="var(--primary)" />
                <span>DataMind AI User Guide & Documentation</span>
              </div>
              <button className="modal-close-btn" onClick={() => setIsDocModalOpen(false)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-tabs custom-scrollbar">
              <button
                className={`modal-tab ${activeDocTab === 'quickstart' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('quickstart')}
              >
                <Compass size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} /> Quick Start
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'eda' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('eda')}
              >
                <BarChart3 size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} /> EDA & Charts
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'overfitting' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('overfitting')}
              >
                <Brain size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} /> Overfitting Guide
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('batch')}
              >
                <Layers size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} /> Batch Predict
              </button>
              <button
                className={`modal-tab ${activeDocTab === 'api' ? 'active' : ''}`}
                onClick={() => setActiveDocTab('api')}
              >
                <Cpu size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} /> API Swagger
              </button>
            </div>

            <div className="modal-body custom-scrollbar">
              {activeDocTab === 'quickstart' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Compass size={20} /> End-to-End DataMind AI Workflow
                  </h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    DataMind AI is an automated data science platform that takes your raw dataset and helps you profile, analyze, train, diagnose, and deploy machine learning models in seconds without writing code.
                  </p>

                  <div style={{ display: 'grid', gap: '1rem' }}>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>1. Upload Dataset (.csv or .xlsx)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Drag & drop your spreadsheet. DataMind profiles row counts, column data types, missing values, and duplicate records instantly.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>2. Run Plain-English EDA (Exploratory Data Analysis)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Produces clear, visual histograms showing value distribution ranges, highlights potential outliers, discovers high correlations, and generates plain-English takeaway insights.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>3. Auto-Train Machine Learning Models</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Select your target variable (e.g. Price, Churn, Species). DataMind automatically cleans data, encodes categories, scales numbers, trains competing algorithms, and picks the champion.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>4. Check Model Diagnostics (Overfitting & Underfitting)</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Review the Train vs Test generalization score chart to verify your model isn’t just memorizing training data.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>5. Batch Predict on New Records</strong>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                        Download the auto-generated CSV template, paste in new unseen records, and run predictions on thousands of rows at once.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'eda' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <BarChart3 size={20} /> How to Read EDA Charts & Numerical Distributions
                  </h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    Exploratory Data Analysis helps you understand what the data looks like before training any algorithm.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.08)', padding: '1rem', borderRadius: '10px', border: '1px solid rgba(99, 102, 241, 0.25)' }}>
                      <strong style={{ color: '#818cf8', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <BarChart2 size={16} /> What is a Numerical Histogram?
                      </strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        Unlike a simple average line, a histogram divides continuous values into 8–10 equal range &quot;bins&quot; along the horizontal X-axis and counts how many rows fall into each range along the vertical Y-axis. The taller the bar, the more records possess values in that bucket.
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <TrendingUp size={16} color="var(--primary)" /> Distribution Shapes Explained
                      </strong>
                      <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                        <li><strong>Normal / Bell-shaped:</strong> Most data clusters symmetrically around the center (mean ≈ median). Ideal for all standard machine learning models.</li>
                        <li><strong>Right-skewed (Tail on Highs):</strong> Most rows are small (e.g. typical salaries or house prices), but a small number of extremely large luxury values stretch out to the right.</li>
                        <li><strong>Left-skewed (Tail on Lows):</strong> Most rows cluster on higher values, with a few unusually small values stretching out to the left.</li>
                      </ul>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <AlertTriangle size={16} color="#fbbf24" /> Outliers (IQR Method)
                      </strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        An outlier is a data point located outside 1.5 times the Interquartile Range (IQR). It could represent a rare real-world occurrence (e.g., a billionaire in an income survey) or a data typo (e.g., Age = 999).
                      </p>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <Activity size={16} color="#818cf8" /> Feature Correlation
                      </strong>
                      <p style={{ color: '#cbd5e1', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
                        Measured from -1.0 to +1.0. A correlation of +0.90 between two columns means that when one increases, the other almost always increases. Having redundant columns can confuse linear models.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'overfitting' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Brain size={20} /> Overfitting vs Underfitting: Plain-English Guide
                  </h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
                    When machine learning models train, they must strike a balance between learning the patterns in the data and generalizing to new, unseen cases.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <h4 style={{ color: '#f87171', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <AlertTriangle size={18} /> What is Overfitting? (High Variance)
                      </h4>
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
                      <h4 style={{ color: '#fbbf24', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <HelpCircle size={18} /> What is Underfitting? (High Bias)
                      </h4>
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
                      <h4 style={{ color: '#34d399', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <CheckCircle2 size={18} /> What is a Good Fit? (Balanced)
                      </h4>
                      <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                        Both Training and Test scores are high and closely matched (Generalization Gap &lt; 5-10%). The model learned generalized patterns that hold true on unseen data.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeDocTab === 'batch' && (
                <div>
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Layers size={20} /> What Does &quot;Batch Predict (Upload CSV)&quot; Mean?
                  </h3>
                  <p style={{ marginBottom: '1.5rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                    In machine learning, training a model is only the first step. The true business value comes when you use that trained model to make predictions on <strong>new records</strong> where the outcome is currently unknown.
                  </p>

                  <div style={{ display: 'grid', gap: '1.25rem' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: '10px', padding: '1.25rem' }}>
                      <strong style={{ color: '#818cf8', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <Layers size={16} /> How Batch Prediction Works:
                      </strong>
                      <ol style={{ paddingLeft: '1.25rem', marginTop: '0.75rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.8' }}>
                        <li>Train a model in DataMind on historical records (e.g. customer churn or house sales).</li>
                        <li>Click <strong>&quot;Download Batch Template&quot;</strong> to receive a clean CSV pre-populated with the exact feature column names your model requires.</li>
                        <li>Fill that CSV with your new rows of customer or product data (without the target column).</li>
                        <li>Click <strong>&quot;Batch Predict (Upload CSV)&quot;</strong> and upload that file.</li>
                        <li>The model runs predictions across all rows, adds a <code>prediction</code> column (and class probabilities), displays the table in the UI, and allows you to download the completed CSV!</li>
                      </ol>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                      <strong style={{ color: 'white' }}>Real-World Use Cases</strong>
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
                  <h3 style={{ marginBottom: '1rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Cpu size={20} /> What Does API Swagger Mean in this Project?
                  </h3>
                  <p style={{ marginBottom: '1.25rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                    <strong>Swagger (OpenAPI)</strong> is an interactive web-based interface automatically generated by FastAPI at <code>/docs</code>. It documents every backend endpoint, accepts test uploads directly in your browser, and serves as the software contract between DataMind and external systems.
                  </p>

                  <div style={{ marginBottom: '1.5rem' }}>
                    <a
                      href={`${API_BASE}/docs`}
                      target="_blank"
                      rel="noreferrer"
                      className="btn btn-primary"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none', fontSize: '0.9rem' }}
                    >
                      <ExternalLink size={16} /> Open Interactive Swagger UI ({API_BASE}/docs)
                    </a>
                  </div>

                  <div style={{ display: 'grid', gap: '1rem', marginBottom: '1.5rem' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.25)', padding: '1rem', borderRadius: '10px' }}>
                      <strong style={{ color: '#818cf8' }}>Why Swagger Matters in the Real World:</strong>
                      <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', color: '#cbd5e1', fontSize: '0.85rem', lineHeight: '1.7' }}>
                        <li><strong>For Business Users:</strong> You can test every backend capability (uploading data, running predictions, chatting with the AI agent) right inside the browser without installing tools like Postman or writing terminal commands.</li>
                        <li><strong>For Developers & Engineers:</strong> If your company already has an ERP, CRM, or mobile app, developers don&apos;t need to use the web frontend—they can connect their software directly to DataMind&apos;s REST API using the exact schemas documented in Swagger.</li>
                        <li><strong>For Automation & Pipelines:</strong> Scheduled batch jobs (e.g. running daily churn predictions every midnight) can call the API endpoints automatically using standard curl, Python, or Node.js requests.</li>
                      </ul>
                    </div>
                  </div>

                  <h4 style={{ color: 'white', marginBottom: '0.75rem', fontSize: '1rem' }}>Key API Endpoints Reference</h4>
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
