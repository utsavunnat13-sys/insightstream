import React, { useState, useRef } from "react";
import {
  Upload,
  Database,
  FileSpreadsheet,
  Trash2,
  ChevronDown,
  ChevronRight,
  Hash,
  Type
} from "lucide-react";

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
}

interface UploadedFile {
  filename: string;
  size_bytes: number;
  cached: boolean;
}

interface SidebarProps {
  files: UploadedFile[];
  activeProfile: DatasetProfile | null;
  onUpload: (file: File) => void;
  onSelectFile: (filename: string) => void;
  onDeleteFile: (filename: string) => void;
  isUploading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  files,
  activeProfile,
  onUpload,
  onSelectFile,
  onDeleteFile,
  isUploading,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [expandedColumn, setExpandedColumn] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith(".csv")) {
        onUpload(file);
      }
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const toggleColumnExpand = (colName: string) => {
    setExpandedColumn(expandedColumn === colName ? null : colName);
  };

  return (
    <div className="flex flex-col h-full bg-[#0b1329]/80 border border-slate-800/80 rounded-2xl overflow-hidden glass-panel p-5 space-y-6">
      {/* App Branding */}
      <div className="flex items-center space-x-3 pb-3 border-b border-slate-800">
        <div className="p-2.5 bg-indigo-600 rounded-xl text-white shadow-lg shadow-indigo-600/20">
          <Database className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-md font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-300">
            InsightStream
          </h1>
          <p className="text-[10px] text-indigo-400 font-semibold tracking-wider uppercase">
            AI Data Workspace
          </p>
        </div>
      </div>

      {/* Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition duration-300 flex flex-col items-center justify-center space-y-2 ${
          dragActive
            ? "border-indigo-500 bg-indigo-500/5"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/20"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={isUploading}
        />
        <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400 border border-indigo-500/20">
          <Upload className="w-5 h-5" />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-200">
            {isUploading ? "Uploading file..." : "Drag & Drop CSV"}
          </p>
          <p className="text-[10px] text-slate-500 mt-1">or click to browse</p>
        </div>
      </div>

      {/* Files List */}
      {files.length > 0 && (
        <div className="space-y-2 flex-1 max-h-[30%] overflow-y-auto pr-1">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Your Files
          </h3>
          <div className="space-y-1.5">
            {files.map((f) => {
              const isActive = activeProfile?.filename === f.filename;
              return (
                <div
                  key={f.filename}
                  onClick={() => onSelectFile(f.filename)}
                  className={`group flex items-center justify-between p-2.5 rounded-xl border transition duration-200 cursor-pointer ${
                    isActive
                      ? "bg-indigo-600/10 border-indigo-500/50 text-slate-100"
                      : "bg-slate-900/30 border-slate-900 hover:bg-slate-800/40 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <div className="flex items-center space-x-2.5 min-w-0">
                    <FileSpreadsheet
                      className={`w-4 h-4 shrink-0 ${
                        isActive ? "text-indigo-400" : "text-slate-500"
                      }`}
                    />
                    <div className="min-w-0">
                      <p className="text-xs font-medium truncate pr-1">
                        {f.filename}
                      </p>
                      <p className="text-[10px] text-slate-500">
                        {formatSize(f.size_bytes)}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteFile(f.filename);
                    }}
                    className="p-1 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-950/20 opacity-0 group-hover:opacity-100 transition duration-200"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Profiler Details */}
      {activeProfile && (
        <div className="flex-1 flex flex-col min-h-0 space-y-2">
          <div className="flex justify-between items-center pb-1">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Data Profile
            </h3>
            <div className="flex items-center space-x-2 text-[10px] text-slate-500">
              <span>{activeProfile.rows.toLocaleString()} rows</span>
              <span>•</span>
              <span>{activeProfile.columns_count} cols</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
            {activeProfile.columns.map((col) => {
              const isExpanded = expandedColumn === col.name;
              return (
                <div
                  key={col.name}
                  className="bg-slate-900/30 border border-slate-900/60 rounded-xl overflow-hidden"
                >
                  {/* Column Header */}
                  <div
                    onClick={() => toggleColumnExpand(col.name)}
                    className="flex items-center justify-between p-2.5 hover:bg-slate-800/20 cursor-pointer transition duration-200"
                  >
                    <div className="flex items-center space-x-2 min-w-0">
                      {col.is_numeric ? (
                        <Hash className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                      ) : (
                        <Type className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      )}
                      <span className="text-xs font-medium text-slate-300 truncate">
                        {col.name}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <span className="text-[10px] text-slate-500 bg-slate-950 border border-slate-900 px-1.5 py-0.5 rounded">
                        {col.type.replace("int64", "num").replace("float64", "num").replace("object", "str")}
                      </span>
                      {isExpanded ? (
                        <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                      )}
                    </div>
                  </div>

                  {/* Expanded stats */}
                  {isExpanded && (
                    <div className="p-3 border-t border-slate-900/80 bg-slate-950/20 text-[10px] text-slate-400 space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-slate-500">Missing Values</p>
                          <p className="text-slate-300 font-medium">
                            {col.null_count} ({col.null_percentage}%)
                          </p>
                        </div>
                        <div>
                          <p className="text-slate-500">Unique Values</p>
                          <p className="text-slate-300 font-medium">
                            {col.unique_count}
                          </p>
                        </div>
                      </div>

                      {col.is_numeric ? (
                        <div className="border-t border-slate-900/60 pt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[9px]">
                          <div>
                            <span className="text-slate-500">Min:</span>{" "}
                            <span className="text-slate-300 font-medium">
                              {col.min ?? "N/A"}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500">Max:</span>{" "}
                            <span className="text-slate-300 font-medium">
                              {col.max ?? "N/A"}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500">Mean:</span>{" "}
                            <span className="text-slate-300 font-medium">
                              {col.mean ?? "N/A"}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500">Std:</span>{" "}
                            <span className="text-slate-300 font-medium">
                              {col.std ?? "N/A"}
                            </span>
                          </div>
                        </div>
                      ) : (
                        col.top_values &&
                        col.top_values.length > 0 && (
                          <div className="border-t border-slate-900/60 pt-2 space-y-1">
                            <p className="text-slate-500 text-[9px] uppercase tracking-wider">
                              Top Values
                            </p>
                            <div className="space-y-1">
                              {col.top_values.map((v) => (
                                <div
                                  key={v.val}
                                  className="flex justify-between text-[9px] bg-slate-900/40 px-1.5 py-0.5 rounded"
                                >
                                  <span className="text-slate-300 truncate pr-2 max-w-[70%]">
                                    {v.val}
                                  </span>
                                  <span className="text-slate-500">
                                    {v.count}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )
                      )}

                      <div className="border-t border-slate-900/60 pt-2">
                        <p className="text-slate-500 text-[9px] uppercase tracking-wider pb-1">
                          Sample Values
                        </p>
                        <p className="text-[9px] text-slate-300 bg-slate-900/40 px-1.5 py-1 rounded leading-normal break-all">
                          {col.sample_values.map(String).join(", ")}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
