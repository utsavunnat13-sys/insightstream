import React, { useState } from "react";
import {
  Sparkles,
  AlertCircle,
  BarChart3,
  Code2,
  Table,
  Clock,
  Search,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  Database,
  ArrowRight,
  Settings,
  Eye,
  EyeOff
} from "lucide-react";
import { Visualizer } from "./Visualizer";

export interface AnalysisResult {
  query: string;
  explanation: string;
  chart_type: string;
  x_key: string | null;
  y_keys: string[] | null;
  data: any[];
  success: boolean;
  python_code?: string;
  has_api_key?: boolean;
}

interface ColumnProfile {
  name: string;
  type: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  sample_values: any[];
  is_numeric: boolean;
  mean?: number | null;
  min?: number | null;
  max?: number | null;
  std?: number | null;
  top_values?: { val: string; count: number }[];
}

interface DatasetProfile {
  filename: string;
  rows: number;
  columns_count: number;
  columns: ColumnProfile[];
  preview?: any[];
}

import type { AppSettings } from "../App";

interface AnalysisPanelProps {
  activeProfile: DatasetProfile | null;
  activeAnalysis: AnalysisResult | null;
  analysisHistory: AnalysisResult[];
  isLoading: boolean;
  onRunQuery: (query: string, mode: "structured" | "rag") => void;
  onSelectHistory: (analysis: AnalysisResult) => void;
  onClearHistory: () => void;
  settings: AppSettings;
  onSaveSettings: (newSettings: AppSettings) => Promise<boolean>;
  isSettingsOpen: boolean;
  setIsSettingsOpen: (isOpen: boolean) => void;
}

const BI_SUGGESTIONS = [
  "What is the average expected salary by department?",
  "Count candidates by status",
  "Years of experience vs expected salary",
  "Show me a summary of the dataset",
];

const RAG_SUGGESTIONS = [
  "Find candidates with strong python skills and engineering background",
  "Search for candidates rejected due to coding performance",
  "Who is the candidate with the highest interview score and what was their status?",
  "Find any applicants mentioning machine learning experience",
];

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  activeProfile,
  activeAnalysis,
  analysisHistory,
  isLoading,
  onRunQuery,
  onSelectHistory,
  onClearHistory,
  settings,
  onSaveSettings,
  isSettingsOpen,
  setIsSettingsOpen
}) => {
  const [query, setQuery] = useState("");
  const [showCode, setShowCode] = useState(false);
  const [mode, setMode] = useState<"structured" | "rag">("structured");
  
  // Local states for the Settings Modal fields
  const [localProvider, setLocalProvider] = useState(settings.provider);
  const [geminiKey, setGeminiKey] = useState(settings.gemini_api_key || "");
  const [geminiModel, setGeminiModel] = useState(settings.gemini_model || "gemini-3.5-flash");
  const [openaiKey, setOpenaiKey] = useState(settings.openai_api_key || "");
  const [openaiModel, setOpenaiModel] = useState(settings.openai_model || "gpt-4o-mini");
  const [anthropicKey, setAnthropicKey] = useState(settings.anthropic_api_key || "");
  const [anthropicModel, setAnthropicModel] = useState(settings.anthropic_model || "claude-3-5-sonnet-latest");
  const [ollamaBase, setOllamaBase] = useState(settings.ollama_api_base || "http://localhost:11434/v1");
  const [ollamaModel, setOllamaModel] = useState(settings.ollama_model || "llama3");

  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Sync settings when opened
  React.useEffect(() => {
    if (isSettingsOpen) {
      setLocalProvider(settings.provider);
      setGeminiKey(settings.gemini_api_key || "");
      setGeminiModel(settings.gemini_model || "gemini-3.5-flash");
      setOpenaiKey(settings.openai_api_key || "");
      setOpenaiModel(settings.openai_model || "gpt-4o-mini");
      setAnthropicKey(settings.anthropic_api_key || "");
      setAnthropicModel(settings.anthropic_model || "claude-3-5-sonnet-latest");
      setOllamaBase(settings.ollama_api_base || "http://localhost:11434/v1");
      setOllamaModel(settings.ollama_model || "llama3");
    }
  }, [isSettingsOpen, settings]);

  const handleSave = async () => {
    setIsSaving(true);
    await onSaveSettings({
      provider: localProvider,
      gemini_api_key: geminiKey,
      gemini_model: geminiModel,
      openai_api_key: openaiKey,
      openai_model: openaiModel,
      anthropic_api_key: anthropicKey,
      anthropic_model: anthropicModel,
      ollama_api_base: ollamaBase,
      ollama_model: ollamaModel,
    });
    setIsSaving(false);
  };

  const activeProvider = settings.provider;
  const isMissingKey = 
    (activeProvider === "gemini" && !settings.gemini_api_key) ||
    (activeProvider === "openai" && !settings.openai_api_key) ||
    (activeProvider === "anthropic" && !settings.anthropic_api_key);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading || !activeProfile) return;
    onRunQuery(query.trim(), mode);
    setQuery("");
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (isLoading || !activeProfile) return;
    onRunQuery(suggestion, mode);
  };

  const renderTable = (data: any[]) => {
    if (!data || data.length === 0) return null;
    const headers = Object.keys(data[0]);

    return (
      <div className="overflow-x-auto border border-slate-800/80 rounded-xl bg-slate-900/10">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/40 text-slate-400 font-semibold uppercase tracking-wider">
              {headers.map((h) => (
                <th key={h} className="p-3">
                  {h.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50 text-slate-300">
            {data.map((row, i) => (
              <tr key={i} className="hover:bg-slate-800/20 transition duration-150">
                {headers.map((h) => (
                  <td key={h} className="p-3 truncate max-w-[200px]">
                    {row[h] === null || row[h] === undefined
                      ? "-"
                      : typeof row[h] === "number"
                      ? Number(row[h]).toLocaleString(undefined, { maximumFractionDigits: 2 })
                      : String(row[h])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Check if API key is present based on the active analysis or default back to checking active profile

  return (
    <div className="flex flex-col h-full bg-[#0b1329]/80 border border-slate-800/80 rounded-2xl overflow-hidden glass-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0f172a]/50 shrink-0">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400 border border-indigo-500/20">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-100">Data Analyzer Workspace</h2>
            <p className="text-xs text-slate-400 flex items-center gap-2">
              {activeProfile ? `Active Dataset: ${activeProfile.filename}` : "Upload a file to start"}
              {activeProfile && (
                <>
                  <span className="text-slate-650">•</span>
                  <span className="text-[10px] text-indigo-400 font-semibold bg-indigo-950/40 border border-indigo-900/60 px-2 py-0.5 rounded-full capitalize flex items-center gap-1">
                    <Sparkles className="w-2.5 h-2.5 text-indigo-400 shrink-0" />
                    <span>Engine: {activeProvider === 'local' ? 'Local Heuristics' : activeProvider}</span>
                  </span>
                </>
              )}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setIsSettingsOpen(true)}
          className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-slate-800 rounded-xl transition duration-200 cursor-pointer flex items-center space-x-2 text-xs font-semibold"
          title="Model Provider Settings"
        >
          <Settings className="w-4 h-4 transition-transform duration-300 hover:rotate-45" />
          <span className="hidden sm:inline">Settings</span>
        </button>
      </div>

      {/* Main Workspace Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 flex flex-col min-h-0">
        {!activeProfile ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6 max-w-md mx-auto">
            <div className="p-4 bg-indigo-600/10 rounded-2xl text-indigo-400 border border-indigo-500/20">
              <Database className="w-12 h-12" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-slate-200">Upload Data to Analyze</h3>
              <p className="text-sm text-slate-400 mt-2 leading-relaxed">
                Upload a CSV dataset using the sidebar to explore columns, run aggregations, and generate visualizations.
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Search/Query Bar & Mode Selector */}
            <div className="space-y-3 shrink-0 flex flex-col">
              <form onSubmit={handleSubmit} className="relative flex items-center">
                <div className="absolute left-4 text-slate-500">
                  <Search className="w-4 h-4" />
                </div>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isLoading}
                  placeholder={
                    mode === "structured"
                      ? "What would you like to analyze? e.g. 'average expected salary by department'..."
                      : "Ask a semantic search/RAG query, e.g. 'candidates with machine learning skills'..."
                  }
                  className="w-full bg-[#0a0f1d] border border-slate-800 focus:border-indigo-500/50 rounded-xl pl-11 pr-24 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none transition duration-200"
                />
                <button
                  type="submit"
                  disabled={!query.trim() || isLoading}
                  className="absolute right-2 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold disabled:bg-slate-800/50 disabled:text-slate-600 transition duration-200 cursor-pointer shadow-md shadow-indigo-600/10"
                >
                  {isLoading ? (mode === "structured" ? "Analyzing..." : "Searching...") : (mode === "structured" ? "Analyze" : "Search")}
                </button>
              </form>

              <div className="flex items-center space-x-2 bg-slate-900/30 p-1 border border-slate-800/50 rounded-xl self-start">
                <button
                  type="button"
                  onClick={() => setMode("structured")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer flex items-center space-x-1.5 ${
                    mode === "structured"
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/15"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" />
                  <span>Structured BI Analysis</span>
                </button>
                <button
                  type="button"
                  onClick={() => setMode("rag")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer flex items-center space-x-1.5 ${
                    mode === "rag"
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/15"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
                  }`}
                >
                  <Database className="w-3.5 h-3.5" />
                  <span>Semantic RAG Search</span>
                </button>
              </div>
            </div>

            {/* API Key Missing Alert */}
            {isMissingKey && (
              <div className="flex items-start space-x-3 text-amber-400 bg-amber-950/15 border border-amber-500/20 px-4 py-3 rounded-xl shrink-0 animate-fade-in text-xs leading-normal">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold">Missing API Key:</span> The selected provider <span className="capitalize font-bold text-indigo-300">{activeProvider}</span> requires an API key to function. Currently running in fallback local heuristic mode. Click the <span className="font-bold cursor-pointer underline text-indigo-400 hover:text-indigo-300" onClick={() => setIsSettingsOpen(true)}>Settings</span> button above to configure your key.
                </div>
              </div>
            )}

            {isLoading && (
              <div className="flex-1 flex flex-col items-center justify-center space-y-3">
                <div className="flex space-x-1.5">
                  <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce delay-100"></div>
                  <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce delay-200"></div>
                  <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce delay-300"></div>
                </div>
                <span className="text-xs text-slate-400 font-medium">
                  {mode === "structured" 
                    ? "Running Pandas aggregation & plotting results..." 
                    : "Generating row embeddings & running semantic similarity search..."}
                </span>
              </div>
            )}

            {!isLoading && !activeAnalysis && (
              <div className="space-y-6">
                {/* Suggestions Grid */}
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Quick Analysis Templates
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {(mode === "structured" ? BI_SUGGESTIONS : RAG_SUGGESTIONS).map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => handleSuggestionClick(s)}
                        className="flex items-center justify-between text-left text-xs bg-slate-900/30 hover:bg-slate-800/40 border border-slate-800/80 rounded-xl px-4 py-3 text-slate-300 hover:text-slate-100 transition duration-200 cursor-pointer group"
                      >
                        <span>{s}</span>
                        <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                      </button>
                    ))}
                  </div>
                </div>

                {/* Raw Dataset Preview */}
                {activeProfile.preview && activeProfile.preview.length > 0 && (
                  <div className="flex-1 flex flex-col min-h-0 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                        <FileSpreadsheet className="w-3.5 h-3.5 text-slate-500" />
                        <span>Dataset Raw Preview (First 10 Rows)</span>
                      </h4>
                      <span className="text-[10px] text-slate-500">
                        {activeProfile.rows.toLocaleString()} rows total
                      </span>
                    </div>
                    <div className="flex-1 overflow-auto border border-slate-800/80 rounded-xl bg-slate-950/20">
                      {renderTable(activeProfile.preview)}
                    </div>
                  </div>
                )}
              </div>
            )}

            {!isLoading && activeAnalysis && (
              <div className="space-y-6">
                {/* Insights and Visualization Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 shrink-0">
                  {/* Insight Panel */}
                  <div className="rounded-xl p-5 flex flex-col space-y-3 glass-panel">
                    <div className="flex items-center space-x-2 text-indigo-400">
                      <Sparkles className="w-4 h-4" />
                      <h4 className="text-xs font-bold uppercase tracking-wider">AI Insights & Explanation</h4>
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
                      {activeAnalysis.explanation}
                    </p>
                  </div>

                  {/* Chart Panel */}
                  {activeAnalysis.chart_type && activeAnalysis.chart_type !== "none" && activeAnalysis.data && (
                    <div className="rounded-xl p-5 flex flex-col glass-panel">
                      <div className="flex items-center space-x-2 text-indigo-400 pb-2">
                        <BarChart3 className="w-4 h-4" />
                        <h4 className="text-xs font-bold uppercase tracking-wider">Data Visualization</h4>
                      </div>
                      <div className="flex-1 chart-wrapper">
                        <Visualizer
                          chartType={activeAnalysis.chart_type}
                          data={activeAnalysis.data}
                          xKey={activeAnalysis.x_key}
                          yKeys={activeAnalysis.y_keys}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Tabular Result Grid */}
                {activeAnalysis.data && activeAnalysis.data.length > 0 && (
                  <div className="flex-1 flex flex-col min-h-[150px] space-y-2">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                      <Table className="w-3.5 h-3.5 text-slate-500" />
                      <span>Result Data Table ({activeAnalysis.data.length} rows returned)</span>
                    </h4>
                    <div className="flex-1 overflow-auto border border-slate-800/80 rounded-xl bg-slate-950/20">
                      {renderTable(activeAnalysis.data)}
                    </div>
                  </div>
                )}

                {/* Collapsible Python Code Block */}
                {activeAnalysis.python_code && (
                  <div className="border border-slate-800/80 rounded-xl overflow-hidden shrink-0">
                    <button
                      type="button"
                      onClick={() => setShowCode(!showCode)}
                      className="w-full flex items-center justify-between px-4 py-3 bg-slate-900/40 hover:bg-slate-900/60 text-slate-400 hover:text-slate-200 transition text-xs font-semibold cursor-pointer"
                    >
                      <span className="flex items-center space-x-2">
                        <Code2 className="w-3.5 h-3.5" />
                        <span>
                          {activeAnalysis.python_code.startsWith("#")
                            ? "View RAG Semantic Retrieval Details"
                            : "View Executed Pandas Python Code"}
                        </span>
                      </span>
                      {showCode ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                    {showCode && (
                      <div className="bg-slate-950/60 p-4 border-t border-slate-800/80 text-left">
                        <pre className="text-xs text-emerald-400/90 font-mono overflow-x-auto whitespace-pre leading-relaxed p-2 bg-[#050b14] rounded-lg border border-slate-900">
                          <code>{activeAnalysis.python_code}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* History Log Panel at the Bottom */}
      {activeProfile && analysisHistory.length > 0 && (
        <div className="border-t border-slate-800/80 bg-[#070c18] px-6 py-4 flex items-center space-x-3 overflow-x-auto shrink-0">
          <div className="flex items-center space-x-1 text-slate-500 text-xs font-semibold uppercase tracking-wider shrink-0 mr-2">
            <Clock className="w-3.5 h-3.5" />
            <span>Recent Queries:</span>
          </div>
          <div className="flex items-center space-x-2 overflow-x-auto py-1">
            {analysisHistory.map((hist, idx) => {
              const isActive = activeAnalysis?.query === hist.query;
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => onSelectHistory(hist)}
                  className={`px-3 py-1.5 rounded-lg text-xs truncate max-w-[200px] border transition cursor-pointer shrink-0 ${
                    isActive
                      ? "bg-indigo-600/10 border-indigo-500/50 text-indigo-300"
                      : "bg-slate-900/40 border-slate-800 hover:bg-slate-800/40 text-slate-400 hover:text-slate-200"
                  }`}
                  title={hist.query}
                >
                  {hist.query}
                </button>
              );
            })}
            <button
              type="button"
              onClick={onClearHistory}
              className="text-[10px] text-slate-500 hover:text-slate-300 font-semibold px-2 py-1 cursor-pointer shrink-0"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="modal-overlay">
          <div className="modal-container">
            {/* Modal Header */}
            <div className="modal-header">
              <div className="flex items-center space-x-2 text-indigo-400">
                <Settings className="w-5 h-5" />
                <h3 className="font-bold text-slate-200">Model Provider Settings</h3>
              </div>
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                className="text-slate-500 hover:text-slate-350 text-xl cursor-pointer"
              >
                &times;
              </button>
            </div>

            {/* Modal Body */}
            <div className="modal-body">
              {/* Select Provider */}
              <div className="form-group">
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Select LLM Provider
                </label>
                <div className="provider-grid">
                  {[
                    { id: "gemini", label: "Gemini", sub: "Google" },
                    { id: "openai", label: "OpenAI", sub: "GPT" },
                    { id: "anthropic", label: "Anthropic", sub: "Claude" },
                    { id: "ollama", label: "Ollama", sub: "Local" },
                    { id: "local", label: "Heuristics", sub: "Offline" },
                  ].map((p) => {
                    const isActive = localProvider === p.id;
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => setLocalProvider(p.id)}
                        className={`flex flex-col items-center justify-center p-2 rounded-xl border text-center transition cursor-pointer ${
                          isActive
                            ? "bg-indigo-600/10 border-indigo-500 text-indigo-300 shadow-md shadow-indigo-600/5"
                            : "bg-slate-900/30 border-slate-800/80 hover:bg-slate-800/30 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <span className="text-xs font-semibold">{p.label}</span>
                        <span className="text-[8px] text-slate-550 mt-0.5">{p.sub}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Gemini Section */}
              {localProvider === "gemini" && (
                <div className="space-y-4 pt-2 border-t border-slate-800/40">
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400 flex items-center justify-between">
                      <span>Gemini API Key</span>
                      <span className="text-[10px] text-slate-500 font-normal">Stored locally</span>
                    </label>
                    <div className="input-wrapper">
                      <input
                        type={showGeminiKey ? "text" : "password"}
                        value={geminiKey}
                        onChange={(e) => setGeminiKey(e.target.value)}
                        placeholder={settings.gemini_api_key ? "•••••••••••••••• (Configured)" : "Enter Gemini API Key..."}
                        className="modal-input"
                      />
                      <button
                        type="button"
                        onClick={() => setShowGeminiKey(!showGeminiKey)}
                        className="password-toggle"
                      >
                        {showGeminiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">Gemini Model</label>
                    <select
                      value={geminiModel}
                      onChange={(e) => setGeminiModel(e.target.value)}
                      className="modal-input"
                    >
                      <option value="gemini-3.5-flash">gemini-3.5-flash (Recommended)</option>
                      <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                      <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                    </select>
                  </div>
                </div>
              )}

              {/* OpenAI Section */}
              {localProvider === "openai" && (
                <div className="space-y-4 pt-2 border-t border-slate-800/40">
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">OpenAI API Key</label>
                    <div className="input-wrapper">
                      <input
                        type={showOpenaiKey ? "text" : "password"}
                        value={openaiKey}
                        onChange={(e) => setOpenaiKey(e.target.value)}
                        placeholder={settings.openai_api_key ? "•••••••••••••••• (Configured)" : "Enter OpenAI API Key (sk-...)"}
                        className="modal-input"
                      />
                      <button
                        type="button"
                        onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                        className="password-toggle"
                      >
                        {showOpenaiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">OpenAI Model</label>
                    <select
                      value={openaiModel}
                      onChange={(e) => setOpenaiModel(e.target.value)}
                      className="modal-input"
                    >
                      <option value="gpt-4o-mini">gpt-4o-mini (Recommended)</option>
                      <option value="gpt-4o">gpt-4o (High Quality)</option>
                      <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Anthropic Section */}
              {localProvider === "anthropic" && (
                <div className="space-y-4 pt-2 border-t border-slate-800/40">
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">Anthropic API Key</label>
                    <div className="input-wrapper">
                      <input
                        type={showAnthropicKey ? "text" : "password"}
                        value={anthropicKey}
                        onChange={(e) => setAnthropicKey(e.target.value)}
                        placeholder={settings.anthropic_api_key ? "•••••••••••••••• (Configured)" : "Enter Anthropic API Key (sk-ant-...)"}
                        className="modal-input"
                      />
                      <button
                        type="button"
                        onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                        className="password-toggle"
                      >
                        {showAnthropicKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">Anthropic Model</label>
                    <select
                      value={anthropicModel}
                      onChange={(e) => setAnthropicModel(e.target.value)}
                      className="modal-input"
                    >
                      <option value="claude-3-5-sonnet-latest">claude-3-5-sonnet (Recommended)</option>
                      <option value="claude-3-5-haiku-latest">claude-3-5-haiku</option>
                      <option value="claude-3-opus-20240229">claude-3-opus</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Ollama Section */}
              {localProvider === "ollama" && (
                <div className="space-y-4 pt-2 border-t border-slate-800/40">
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">Ollama API Base URL</label>
                    <input
                      type="text"
                      value={ollamaBase}
                      onChange={(e) => setOllamaBase(e.target.value)}
                      placeholder="e.g. http://localhost:11434/v1"
                      className="modal-input"
                    />
                    <p className="text-[10px] text-slate-550">Ensure Ollama is running locally and CORS is configured.</p>
                  </div>
                  <div className="form-group">
                    <label className="text-xs font-semibold text-slate-400">Model Name</label>
                    <input
                      type="text"
                      value={ollamaModel}
                      onChange={(e) => setOllamaModel(e.target.value)}
                      placeholder="e.g. llama3, mistral, codellama"
                      className="modal-input"
                    />
                  </div>
                </div>
              )}

              {/* Local Heuristics Section */}
              {localProvider === "local" && (
                <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl pt-2 mt-2 space-y-2">
                  <h4 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Local Heuristic Engine</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Uses a built-in offline rules engine to parse questions and run Pandas queries without making any API calls.
                  </p>
                  <p className="text-[10px] text-amber-500/80 italic">
                    Note: Complex natural language reasoning may be limited under local heuristics compared to LLMs.
                  </p>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="modal-footer">
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                disabled={isSaving}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={isSaving}
                className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition cursor-pointer shadow-md shadow-indigo-600/20 disabled:bg-slate-800/50 disabled:text-slate-650"
              >
                {isSaving ? "Saving..." : "Save Settings"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
