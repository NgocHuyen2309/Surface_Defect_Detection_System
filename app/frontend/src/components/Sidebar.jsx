import React from 'react';
import { Activity, BarChart2, Settings, ShieldAlert } from 'lucide-react';
import { cn } from '../lib/utils';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'live', icon: Activity, label: 'Live Inspection' },
    { id: 'analytics', icon: BarChart2, label: 'System Analytics' },
    { id: 'settings', icon: Settings, label: 'Settings' }
  ];

  return (
    <div className="w-64 bg-surface border-r border-border h-screen flex flex-col fixed left-0 top-0">
      <div className="p-6 flex items-center gap-3 border-b border-border">
        <div className="w-10 h-10 bg-primary-50 rounded-xl flex items-center justify-center text-primary-600">
          <ShieldAlert size={24} />
        </div>
        <div>
          <h1 className="font-bold text-gray-900 leading-tight">FabricAI</h1>
          <p className="text-xs text-text-muted font-medium">Defect Detection</p>
        </div>
      </div>

      <div className="flex-1 py-6 px-4 flex flex-col gap-2">
        <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 px-3">
          Menu
        </div>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                isActive 
                  ? "bg-primary-50 text-primary-700" 
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              )}
            >
              <Icon size={18} className={isActive ? "text-primary-600" : "text-gray-400"} />
              {item.label}
            </button>
          );
        })}
      </div>

      <div className="p-4 border-t border-border">
        <div className="bg-gray-50 rounded-lg p-4 flex flex-col items-center text-center">
          <div className="w-2 h-2 rounded-full bg-success-500 mb-2 animate-pulse"></div>
          <p className="text-xs font-medium text-gray-900">System Status: Online</p>
          <p className="text-[10px] text-gray-500 mt-1">Backend Connection Active</p>
        </div>
      </div>
    </div>
  );
}
