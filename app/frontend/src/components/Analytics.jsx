import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { Zap, Target, Award, Activity } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'https://fabric-ai-proxy.eastasia.cloudapp.azure.com';

export default function Analytics() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/history`);
        setHistory(res.data);
      } catch (err) {
        console.error("Failed to fetch history:", err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchHistory();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchHistory, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading && history.length === 0) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="w-10 h-10 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
      </div>
    );
  }

  // 1. Process Data for KPI Cards
  const totalPredictions = history.length;
  const overallAvgConf = totalPredictions > 0 
    ? (history.reduce((acc, curr) => acc + curr.confidence_score, 0) / totalPredictions * 100).toFixed(1)
    : 0;

  const peakFps = totalPredictions > 0
    ? Math.max(...history.map(item => item.inference_time_ms > 0 ? 1000/item.inference_time_ms : 0)).toFixed(1)
    : 0;

  // 2. Process Data for Bar Chart (Averages by Pipeline)
  const pipelineStats = {};

  history.forEach(item => {
    let fps = item.inference_time_ms > 0 ? 1000 / item.inference_time_ms : 0;
    const pName = item.pipeline_used || 'unknown';
    if (!pipelineStats[pName]) {
      pipelineStats[pName] = { count: 0, confSum: 0, fpsSum: 0 };
    }
    pipelineStats[pName].count++;
    pipelineStats[pName].confSum += item.confidence_score;
    pipelineStats[pName].fpsSum += fps;
  });

  const chartData = Object.keys(pipelineStats).map(key => {
    // Make it look pretty: morphological (rf) -> Morphological (RF)
    const formattedName = key.replace(/\b\w/g, c => c.toUpperCase()).replace('(rf)', '(RF)').replace('(svm)', '(SVM)');
    return {
      name: formattedName,
      avg_confidence: parseFloat((pipelineStats[key].confSum / pipelineStats[key].count * 100).toFixed(1)),
      avg_fps: parseFloat((pipelineStats[key].fpsSum / pipelineStats[key].count).toFixed(1))
    };
  });

  // 3. Process Data for Line Chart (Recent Timeline)
  const timelineData = [...history].slice(0, 15).reverse().map((item, idx) => ({
    name: `#${idx + 1}`,
    fps: item.inference_time_ms > 0 ? parseFloat((1000 / item.inference_time_ms).toFixed(1)) : 0,
  }));

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">System Analytics</h2>
        <p className="text-text-muted mt-1">Real-time Performance & Prediction History</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-5 bg-gradient-to-br from-primary-500 to-primary-600 text-white border-none">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-primary-100 text-xs font-medium uppercase tracking-wider mb-1">Avg Confidence</p>
              <h3 className="text-3xl font-bold">{overallAvgConf}%</h3>
            </div>
            <div className="p-2 bg-white/20 rounded-lg">
              <Award size={20} className="text-white" />
            </div>
          </div>
          <p className="text-sm text-primary-100 mt-4 font-medium">Across all pipelines</p>
        </div>

        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">Peak Inference Speed</p>
              <h3 className="text-3xl font-bold text-gray-900">
                {peakFps} <span className="text-sm font-normal text-gray-500">FPS</span>
              </h3>
            </div>
            <div className="p-2 bg-success-50 rounded-lg">
              <Zap size={20} className="text-success-500" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium">Fastest recorded request</p>
        </div>

        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">Total Predictions</p>
              <h3 className="text-3xl font-bold text-gray-900">{totalPredictions}</h3>
            </div>
            <div className="p-2 bg-blue-50 rounded-lg">
              <Target size={20} className="text-blue-500" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium">Images processed</p>
        </div>
        
        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">System Health</p>
              <h3 className="text-3xl font-bold text-gray-900">Live</h3>
            </div>
            <div className="p-2 bg-gray-100 rounded-lg">
              <Activity size={20} className="text-gray-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-success-500 animate-pulse"></span> DB Connected
          </p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confidence Chart */}
        <div className="card p-5">
          <h3 className="text-base font-bold text-gray-900 mb-6">Avg Confidence by Pipeline</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height={288}>
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <RechartsTooltip 
                  cursor={{fill: '#f3f4f6'}}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="avg_confidence" name="Avg Confidence (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speed Chart */}
        <div className="card p-5">
          <h3 className="text-base font-bold text-gray-900 mb-6">Recent Inference Speed (FPS)</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height={288}>
              <LineChart data={timelineData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <RechartsTooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Line type="monotone" dataKey="fps" name="Frames Per Second" stroke="#10b981" strokeWidth={3} dot={{ r: 4, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* History Table */}
      <div className="card overflow-hidden">
        <div className="p-5 border-b border-border">
          <h3 className="text-base font-bold text-gray-900">Recent Prediction Log</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-gray-500 uppercase bg-gray-50/50">
              <tr>
                <th className="px-6 py-4 font-semibold">Time</th>
                <th className="px-6 py-4 font-semibold">Image</th>
                <th className="px-6 py-4 font-semibold">Pipeline</th>
                <th className="px-6 py-4 font-semibold">Defect Type</th>
                <th className="px-6 py-4 font-semibold text-right">Confidence</th>
                <th className="px-6 py-4 font-semibold text-right">Speed</th>
              </tr>
            </thead>
            <tbody>
              {history.slice(0, 10).map((row) => (
                <tr key={row.id} className="border-b border-gray-100 last:border-0 hover:bg-gray-50/50 transition-colors">
                  <td className="px-6 py-4 text-gray-600 whitespace-nowrap">
                    {new Date(row.created_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4">
                    <img 
                      src={`${API_BASE_URL}${row.original_image_url}`} 
                      alt="Thumbnail" 
                      className="w-12 h-12 object-cover rounded shadow-sm border border-gray-200"
                    />
                  </td>
                  <td className="px-6 py-4 font-medium text-gray-900 capitalize">{row.pipeline_used}</td>
                  <td className="px-6 py-4 text-gray-600">
                    <span className={row.defect_type !== 'Good' ? 'text-danger-600 font-semibold' : 'text-success-600 font-semibold'}>
                      {row.defect_type}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="font-semibold text-gray-900">
                      {(row.confidence_score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-gray-600">
                    {row.inference_time_ms}ms
                  </td>
                </tr>
              ))}
              
              {history.length === 0 && (
                <tr>
                  <td colSpan="6" className="px-6 py-10 text-center text-gray-500">
                    No prediction history found. Go to Live Inspection to analyze some images.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
