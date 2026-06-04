import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { mockData } from '../data/mockData';
import { Zap, Target, Award, Activity } from 'lucide-react';

export default function Analytics() {
  const { ab_testing_stats } = mockData;

  // Prepare data for charts
  const chartData = ab_testing_stats.map(item => ({
    name: `${item.pipeline} + ${item.model}`,
    f1_score: item.f1_score * 100, // percentage for better display
    fps: item.fps,
    pipeline: item.pipeline,
    model: item.model
  }));

  const bestModel = ab_testing_stats.reduce((prev, current) => 
    (prev.f1_score > current.f1_score) ? prev : current
  );

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">System Analytics</h2>
        <p className="text-text-muted mt-1">A/B Testing Report: Morphological vs Canny Pipeline</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-5 bg-gradient-to-br from-primary-500 to-primary-600 text-white border-none">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-primary-100 text-xs font-medium uppercase tracking-wider mb-1">Top Accuracy (F1)</p>
              <h3 className="text-3xl font-bold">{bestModel.f1_score * 100}%</h3>
            </div>
            <div className="p-2 bg-white/20 rounded-lg">
              <Award size={20} className="text-white" />
            </div>
          </div>
          <p className="text-sm text-primary-100 mt-4 font-medium">{bestModel.pipeline} + {bestModel.model}</p>
        </div>

        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">Peak Inference Speed</p>
              <h3 className="text-3xl font-bold text-gray-900">
                {Math.max(...ab_testing_stats.map(s => s.fps))} <span className="text-sm font-normal text-gray-500">FPS</span>
              </h3>
            </div>
            <div className="p-2 bg-success-50 rounded-lg">
              <Zap size={20} className="text-success-500" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium">Canny + Random Forest</p>
        </div>

        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">Total Configurations</p>
              <h3 className="text-3xl font-bold text-gray-900">{ab_testing_stats.length}</h3>
            </div>
            <div className="p-2 bg-blue-50 rounded-lg">
              <Target size={20} className="text-blue-500" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium">Active models tested</p>
        </div>
        
        <div className="card p-5">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-text-muted text-xs font-medium uppercase tracking-wider mb-1">System Health</p>
              <h3 className="text-3xl font-bold text-gray-900">Stable</h3>
            </div>
            <div className="p-2 bg-gray-100 rounded-lg">
              <Activity size={20} className="text-gray-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4 font-medium flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-success-500"></span> Backend Connected
          </p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* F1 Score Chart */}
        <div className="card p-5">
          <h3 className="text-base font-bold text-gray-900 mb-6">F1-Score Comparison</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <RechartsTooltip 
                  cursor={{fill: '#f3f4f6'}}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="f1_score" name="F1 Score (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speed Chart */}
        <div className="card p-5">
          <h3 className="text-base font-bold text-gray-900 mb-6">Inference Speed (FPS)</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <RechartsTooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Line type="monotone" dataKey="fps" name="Frames Per Second" stroke="#10b981" strokeWidth={3} dot={{ r: 6, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cross-comparison Table */}
      <div className="card overflow-hidden">
        <div className="p-5 border-b border-border">
          <h3 className="text-base font-bold text-gray-900">Performance Matrix</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-gray-500 uppercase bg-gray-50/50">
              <tr>
                <th className="px-6 py-4 font-semibold">Pipeline</th>
                <th className="px-6 py-4 font-semibold">Model</th>
                <th className="px-6 py-4 font-semibold text-right">F1-Score</th>
                <th className="px-6 py-4 font-semibold text-right">FPS</th>
                <th className="px-6 py-4 font-semibold text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {ab_testing_stats.map((row, idx) => (
                <tr key={idx} className="border-b border-gray-100 last:border-0 hover:bg-gray-50/50 transition-colors">
                  <td className="px-6 py-4 font-medium text-gray-900">{row.pipeline}</td>
                  <td className="px-6 py-4 text-gray-600">{row.model}</td>
                  <td className="px-6 py-4 text-right">
                    <span className={`font-semibold ${row.f1_score > 0.8 ? 'text-success-600' : row.f1_score > 0.5 ? 'text-warning-600' : 'text-danger-600'}`}>
                      {row.f1_score.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-gray-600">{row.fps}</td>
                  <td className="px-6 py-4 text-center">
                    {row.f1_score === bestModel.f1_score ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-success-50 text-success-700">
                        <Award size={12} /> Recommended
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                        Evaluated
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
