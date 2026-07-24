import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

interface VisualizerProps {
  chartType: string;
  data: any[];
  xKey?: string | null;
  yKeys?: string[] | null;
}

const COLORS = [
  "#6366f1", // Indigo
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#3b82f6", // Blue
  "#ec4899", // Pink
  "#8b5cf6", // Purple
];

const GRADIENTS = [
  { id: "gradIndigo", start: "#818cf8", end: "#4f46e5" },
  { id: "gradEmerald", start: "#34d399", end: "#059669" },
  { id: "gradAmber", start: "#fbbf24", end: "#d97706" },
  { id: "gradBlue", start: "#60a5fa", end: "#2563eb" },
];

export const Visualizer: React.FC<VisualizerProps> = ({
  chartType,
  data,
  xKey,
  yKeys,
}) => {
  if (!data || data.length === 0 || chartType === "none") {
    return null;
  }

  // Fallbacks if keys are not provided
  const resolvedXKey = xKey || Object.keys(data[0])[0];
  const resolvedYKeys = yKeys && yKeys.length > 0 ? yKeys : [Object.keys(data[0])[1]];

  const renderChart = () => {
    switch (chartType) {
      case "bar":
        return (
          <BarChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
            <defs>
              {GRADIENTS.map((g) => (
                <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={g.start} stopOpacity={0.9} />
                  <stop offset="95%" stopColor={g.end} stopOpacity={0.6} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
            <XAxis
              dataKey={resolvedXKey}
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              dx={-5}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(17, 24, 39, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
            {resolvedYKeys.map((yKey, index) => {
              const gradId = GRADIENTS[index % GRADIENTS.length].id;
              return (
                <Bar
                  key={yKey}
                  dataKey={yKey}
                  fill={`url(#${gradId})`}
                  radius={[4, 4, 0, 0]}
                  maxBarSize={50}
                />
              );
            })}
          </BarChart>
        );

      case "line":
        return (
          <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
            <XAxis
              dataKey={resolvedXKey}
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              dx={-5}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(17, 24, 39, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
            {resolvedYKeys.map((yKey, index) => (
              <Line
                key={yKey}
                type="monotone"
                dataKey={yKey}
                stroke={COLORS[index % COLORS.length]}
                strokeWidth={3}
                dot={{ r: 4, strokeWidth: 1 }}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        );

      case "pie":
        // For pie, we assume resolvedXKey is the label and resolvedYKeys[0] is the value
        const valKey = resolvedYKeys[0];
        return (
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={true}
              label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey={valKey}
              nameKey={resolvedXKey}
            >
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(17, 24, 39, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
          </PieChart>
        );

      case "scatter":
        const yValKey = resolvedYKeys[0];
        return (
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" />
            <XAxis
              type="number"
              dataKey={resolvedXKey}
              name={resolvedXKey}
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
            />
            <YAxis
              type="number"
              dataKey={yValKey}
              name={yValKey}
              stroke="rgba(255, 255, 255, 0.6)"
              fontSize={12}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{
                backgroundColor: "rgba(17, 24, 39, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
            <Scatter name={`${resolvedXKey} vs ${yValKey}`} data={data} fill="#6366f1" />
          </ScatterChart>
        );

      default:
        return null;
    }
  };

  return (
    <div className="w-full rounded-xl p-4 mt-4 glass-panel chart-container">
      <ResponsiveContainer width="100%" height="100%">
        {renderChart() || <div>Unsupported Chart Type</div>}
      </ResponsiveContainer>
    </div>
  );
};
