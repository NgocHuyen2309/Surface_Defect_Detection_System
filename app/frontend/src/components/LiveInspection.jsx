import React, { useState } from 'react';
import { UploadCloud, CheckCircle2, AlertTriangle, Layers, Cpu, Settings2, Image as ImageIcon } from 'lucide-react';
import BeforeAfterSlider from './BeforeAfterSlider';
import { mockData } from '../data/mockData';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

export default function LiveInspection() {
  const [pipeline, setPipeline] = useState('morphological');
  const [isUploading, setIsUploading] = useState(false);
  const [hasResult, setHasResult] = useState(true); // Default to true to show mock data UI

  const { prediction, images } = mockData;

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Live Inspection</h2>
          <p className="text-text-muted mt-1">Upload fabric image and run real-time defect detection.</p>
        </div>
        
        <div className="flex items-center gap-2 bg-surface p-1 rounded-xl border border-border shadow-sm">
          <button 
            onClick={() => setPipeline('morphological')}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
              pipeline === 'morphological' 
                ? "bg-primary-50 text-primary-700 shadow-sm" 
                : "text-text-muted hover:text-gray-900"
            )}
          >
            Morphological (Recommended)
          </button>
          <button 
            onClick={() => setPipeline('canny')}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
              pipeline === 'canny' 
                ? "bg-primary-50 text-primary-700 shadow-sm" 
                : "text-text-muted hover:text-gray-900"
            )}
          >
            Canny Edge
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Upload & Pipeline Steps */}
        <div className="lg:col-span-1 space-y-6">
          {/* Upload Card */}
          <div className="card p-6 border-dashed border-2 border-gray-300 bg-gray-50/50 hover:bg-gray-50 hover:border-primary-400 transition-colors cursor-pointer group flex flex-col items-center justify-center text-center h-48">
            <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
              <UploadCloud className="text-primary-500" size={24} />
            </div>
            <p className="font-semibold text-gray-800">Upload Fabric Image</p>
            <p className="text-xs text-text-muted mt-1">Drag and drop or click to browse</p>
          </div>

          {/* Pipeline Viewer */}
          <div className="card p-5">
            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-500 mb-4 flex items-center gap-2">
              <Settings2 size={16} /> Explainable AI Pipeline
            </h3>
            
            <div className="space-y-4 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-gray-200 before:to-transparent">
              {/* Step 1 */}
              <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-primary-100 text-primary-600 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2">
                  <ImageIcon size={18} />
                </div>
                <div className="w-[calc(100%-3rem)] md:w-[calc(50%-2.5rem)] p-3 rounded-lg border border-border bg-white shadow-sm ml-4 md:ml-0">
                  <p className="text-xs font-bold text-gray-900">Original Image</p>
                  <p className="text-[10px] text-gray-500 mt-1">Raw fabric capture</p>
                </div>
              </div>
              
              {/* Step 2 */}
              <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-primary-100 text-primary-600 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2">
                  <Layers size={18} />
                </div>
                <div className="w-[calc(100%-3rem)] md:w-[calc(50%-2.5rem)] p-3 rounded-lg border border-border bg-white shadow-sm ml-4 md:ml-0">
                  <p className="text-xs font-bold text-gray-900">Preprocessing</p>
                  <p className="text-[10px] text-gray-500 mt-1">Median Filter + TopHat</p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-primary-100 text-primary-600 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2">
                  <Cpu size={18} />
                </div>
                <div className="w-[calc(100%-3rem)] md:w-[calc(50%-2.5rem)] p-3 rounded-lg border border-border bg-white shadow-sm ml-4 md:ml-0">
                  <p className="text-xs font-bold text-gray-900">Segmentation</p>
                  <p className="text-[10px] text-gray-500 mt-1">Binary Mask ({pipeline === 'morphological' ? 'Morph' : 'Canny'})</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visualizer & Results */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Main Visualizer */}
          <div className="card p-2 bg-white">
            <BeforeAfterSlider 
              beforeImage={images.original} 
              afterImage={images.final_overlay} 
            />
          </div>

          {/* Prediction Results */}
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-4"
          >
            {/* Status Card */}
            <div className={cn(
              "card p-5 border-l-4", 
              prediction.status.includes('DEFECT') ? "border-l-danger-500" : "border-l-success-500"
            )}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-medium text-text-muted uppercase tracking-wide">Analysis Status</p>
                  <h3 className={cn(
                    "text-xl font-bold mt-1",
                    prediction.status.includes('DEFECT') ? "text-danger-600" : "text-success-600"
                  )}>
                    {prediction.status}
                  </h3>
                  {prediction.status.includes('DEFECT') && (
                    <p className="text-sm font-medium text-gray-900 mt-1">
                      Type: <span className="text-danger-500 bg-danger-50 px-2 py-0.5 rounded ml-1">{prediction.defect_type}</span>
                    </p>
                  )}
                </div>
                {prediction.status.includes('DEFECT') 
                  ? <AlertTriangle className="text-danger-500" size={28} />
                  : <CheckCircle2 className="text-success-500" size={28} />
                }
              </div>
              
              <div className="mt-5">
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="font-medium text-gray-700">Confidence Score</span>
                  <span className="font-bold text-gray-900">{prediction.confidence * 100}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-2">
                  <div 
                    className={cn(
                      "h-2 rounded-full",
                      prediction.confidence > 0.9 ? "bg-success-500" : "bg-warning-500"
                    )} 
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Metrics Card */}
            <div className="card p-5">
              <p className="text-xs font-medium text-text-muted uppercase tracking-wide mb-4">Extracted Features</p>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                  <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Max Area</p>
                  <p className="text-lg font-bold text-gray-900">{prediction.features_extracted.max_area} <span className="text-xs font-normal text-gray-500">px²</span></p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                  <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Eccentricity</p>
                  <p className="text-lg font-bold text-gray-900">{prediction.features_extracted.min_eccentricity}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 col-span-2 flex justify-between items-center">
                  <div>
                    <p className="text-[10px] text-gray-500 font-medium uppercase mb-0.5">Pipeline</p>
                    <p className="text-sm font-semibold text-gray-800">{prediction.pipeline_used}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] text-gray-500 font-medium uppercase mb-0.5">Time</p>
                    <p className="text-sm font-bold text-primary-600">{prediction.inference_time}</p>
                  </div>
                </div>
              </div>
            </div>
            
          </motion.div>
        </div>
      </div>
    </div>
  );
}
