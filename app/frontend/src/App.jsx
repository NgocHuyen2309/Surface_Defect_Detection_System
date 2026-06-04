import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import LiveInspection from './components/LiveInspection';
import Analytics from './components/Analytics';

function App() {
  const [activeTab, setActiveTab] = useState('live');

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 ml-64 p-8 overflow-y-auto h-screen">
        <div className="max-w-7xl mx-auto">
          {activeTab === 'live' && <LiveInspection />}
          {activeTab === 'analytics' && <Analytics />}
          {activeTab === 'settings' && (
            <div className="flex items-center justify-center h-full text-text-muted mt-20">
              Settings Panel (Not implemented)
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
