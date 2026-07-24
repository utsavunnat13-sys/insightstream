import { useState, useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { AnalysisPanel } from "./components/AnalysisPanel";
import type { AnalysisResult } from "./components/AnalysisPanel";
import { AlertCircle } from "lucide-react";

export interface AppSettings {
  provider: string;
  gemini_api_key: string | null;
  gemini_model: string | null;
  openai_api_key: string | null;
  openai_model: string | null;
  anthropic_api_key: string | null;
  anthropic_model: string | null;
  ollama_api_base: string | null;
  ollama_model: string | null;
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

interface UploadedFile {
  filename: string;
  size_bytes: number;
  cached: boolean;
}

const API_BASE = import.meta.env.DEV ? "http://localhost:8000/api" : "/api";

function App() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [activeProfile, setActiveProfile] = useState<DatasetProfile | null>(null);
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [settings, setSettings] = useState<AppSettings>({
    provider: "local",
    gemini_api_key: "",
    gemini_model: "gemini-3.5-flash",
    openai_api_key: "",
    openai_model: "gpt-4o-mini",
    anthropic_api_key: "",
    anthropic_model: "claude-3-5-sonnet-latest",
    ollama_api_base: "http://localhost:11434/v1",
    ollama_model: "llama3",
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Fetch file list and settings on mount
  useEffect(() => {
    fetchFiles();
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await fetch(`${API_BASE}/settings`);
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (e) {
      console.error("Failed to fetch settings:", e);
    }
  };

  const handleSaveSettings = async (newSettings: AppSettings) => {
    try {
      const response = await fetch(`${API_BASE}/settings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newSettings),
      });
      if (response.ok) {
        setSettings(newSettings);
        setIsSettingsOpen(false);
        return true;
      } else {
        const err = await response.json();
        throw new Error(err.detail || "Failed to save settings");
      }
    } catch (e: any) {
      console.error("Failed to save settings:", e);
      setError(e.message || "Failed to save settings");
      return false;
    }
  };

  const fetchFiles = async () => {
    try {
      const response = await fetch(`${API_BASE}/files`);
      if (response.ok) {
        const data = await response.json();
        setFiles(data);
      }
    } catch (e) {
      console.error("Failed to fetch files:", e);
      setError("Unable to connect to the backend server. Make sure it is running on port 8000.");
    }
  };

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to upload file");
      }

      const profile: DatasetProfile = await response.json();
      setActiveProfile(profile);
      // Clear analysis workspace for new file
      setActiveAnalysis(null);
      setAnalysisHistory([]);
      fetchFiles();
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to upload file");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelectFile = async (filename: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/files/${filename}`);
      if (!response.ok) {
        throw new Error("Failed to load file profile");
      }
      const profile: DatasetProfile = await response.json();
      setActiveProfile(profile);
      // Reset workspace for selected file
      setActiveAnalysis(null);
      setAnalysisHistory([]);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to select file");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteFile = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE}/files/${filename}`, {
        method: "DELETE",
      });
      if (response.ok) {
        if (activeProfile?.filename === filename) {
          setActiveProfile(null);
          setActiveAnalysis(null);
          setAnalysisHistory([]);
        }
        fetchFiles();
      }
    } catch (e) {
      console.error("Failed to delete file:", e);
    }
  };

  const handleRunQuery = async (queryText: string, mode: "structured" | "rag" = "structured") => {
    if (!activeProfile) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: queryText,
          filename: activeProfile.filename,
          mode: mode,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to query dataset");
      }

      const result = await response.json();

      const newAnalysis: AnalysisResult = {
        query: queryText,
        explanation: result.explanation,
        chart_type: result.chart_type,
        x_key: result.x_key,
        y_keys: result.y_keys,
        data: result.data,
        success: result.success,
        python_code: result.python_code,
        has_api_key: result.has_api_key,
      };

      setActiveAnalysis(newAnalysis);
      setAnalysisHistory((prev) => {
        const filtered = prev.filter((p) => p.query.toLowerCase() !== queryText.toLowerCase());
        return [newAnalysis, ...filtered];
      });
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to run query");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#030712] text-slate-100 overflow-hidden font-sans">
      {/* Background radial glow */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(99,102,241,0.12),rgba(255,255,255,0))] pointer-events-none"></div>

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden p-4 gap-4 z-10">
        {/* Left Side: Sidebar Workspace */}
        <div className="w-80 shrink-0 h-full">
          <Sidebar
            files={files}
            activeProfile={activeProfile}
            onUpload={handleUpload}
            onSelectFile={handleSelectFile}
            onDeleteFile={handleDeleteFile}
            isUploading={isUploading}
          />
        </div>

        {/* Right Side: Analyzer Panel */}
        <div className="flex-1 flex flex-col h-full min-w-0">
          {error && (
            <div className="flex items-center space-x-2 text-red-400 bg-red-950/20 border border-red-500/25 px-4 py-3 rounded-xl mb-4 shrink-0 animate-fade-in">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span className="text-xs">{error}</span>
            </div>
          )}

          <div className="flex-1 flex flex-col min-h-0">
            <AnalysisPanel
              activeProfile={activeProfile}
              activeAnalysis={activeAnalysis}
              analysisHistory={analysisHistory}
              isLoading={isLoading}
              onRunQuery={handleRunQuery}
              onSelectHistory={(analysis) => setActiveAnalysis(analysis)}
              onClearHistory={() => {
                setAnalysisHistory([]);
                setActiveAnalysis(null);
              }}
              settings={settings}
              onSaveSettings={handleSaveSettings}
              isSettingsOpen={isSettingsOpen}
              setIsSettingsOpen={setIsSettingsOpen}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
