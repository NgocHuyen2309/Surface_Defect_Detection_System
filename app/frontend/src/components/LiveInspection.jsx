import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertTriangle, Layers, Cpu, Settings2, Image as ImageIcon } from 'lucide-react';
import BeforeAfterSlider from './BeforeAfterSlider';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';
import axios from 'axios';

// The Backend base URL (assuming user runs Node.js on port 3000 locally)
const API_BASE_URL = 'http://localhost:3000';

export default function LiveInspection() {
  const [pipeline, setPipeline] = useState('morphological');
  const [modelType, setModelType] = useState('rf');
  const [isUploading, setIsUploading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleCardClick = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    setCurrentFile(file); // Triggers useEffect below
    
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  React.useEffect(() => {
    const runPrediction = async () => {
      if (!currentFile) return;
      
      setResultData(null);
      setIsUploading(true);

      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('pipeline_type', pipeline);
      formData.append('model_type', modelType);

      try {
        const response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResultData(response.data.data);
      } catch (error) {
        console.error("Upload error:", error);
        alert(error?.response?.data?.error || "Failed to analyze image. Ensure both backend services are running.");
      } finally {
        setIsUploading(false);
      }
    };

    runPrediction();
  }, [currentFile, pipeline, modelType]);

  const hasResult = resultData !== null;
  const statusStr = resultData?.defect_type === "Good" ? "NO DEFECT" : "DEFECT DETECTED";
  const isDefect = resultData?.defect_type !== "Good";

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Live Inspection</h2>
          <p className="text-text-muted mt-1">Upload fabric image and run real-time defect detection.</p>
        </div>
        
        <div className="flex flex-col gap-2 mb-6">
          <div className="flex bg-gray-100 p-1 rounded-lg">
            <button 
              className={cn(
                "flex-1 px-4 py-1.5 text-sm font-semibold rounded-md transition-all duration-200",
                pipeline === 'morphological' 
                  ? "bg-white text-primary-700 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
              )}
              onClick={() => setPipeline('morphological')}
            >
              Morphological
            </button>
            <button 
              className={cn(
                "flex-1 px-4 py-1.5 text-sm font-semibold rounded-md transition-all duration-200",
                pipeline === 'directional' 
                  ? "bg-white text-primary-700 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
              )}
              onClick={() => setPipeline('directional')}
            >
              Directional Gradient
            </button>
          </div>
          
          <div className="flex bg-gray-100 p-1 rounded-lg w-full max-w-[300px] ml-auto">
            <button 
              className={cn(
                "flex-1 px-3 py-1 text-xs font-semibold rounded-md transition-all duration-200",
                modelType === 'rf' 
                  ? "bg-white text-indigo-700 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
              )}
              onClick={() => setModelType('rf')}
            >
              Random Forest
            </button>
            <button 
              className={cn(
                "flex-1 px-3 py-1 text-xs font-semibold rounded-md transition-all duration-200",
                modelType === 'svm' 
                  ? "bg-white text-indigo-700 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
              )}
              onClick={() => setModelType('svm')}
            >
              SVM
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Upload & Pipeline Steps */}
        <div className="lg:col-span-1 space-y-6">
          {/* Upload Card */}
          <div 
            onClick={handleCardClick}
            className={cn(
              "card p-6 border-dashed border-2 bg-gray-50/50 transition-all group flex flex-col items-center justify-center text-center h-48 relative overflow-hidden",
              isUploading ? "border-primary-300 cursor-wait" : "border-gray-300 hover:bg-gray-50 hover:border-primary-400 cursor-pointer"
            )}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/*"
              onChange={handleFileUpload} 
            />
            {isUploading ? (
              <div className="flex flex-col items-center justify-center z-10 bg-white/80 absolute inset-0">
                <div className="w-10 h-10 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mb-3"></div>
                <p className="font-semibold text-primary-700">Analyzing Image...</p>
              </div>
            ) : previewUrl ? (
              <div className="absolute inset-0">
                <img src={previewUrl} alt="Preview" className="w-full h-full object-cover opacity-30 group-hover:opacity-20 transition-opacity" />
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-3">
                    <UploadCloud className="text-primary-500" size={24} />
                  </div>
                  <p className="font-semibold text-gray-900">Upload New Image</p>
                </div>
              </div>
            ) : (
              <>
                <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                  <UploadCloud className="text-primary-500" size={24} />
                </div>
                <p className="font-semibold text-gray-800">Upload Fabric Image</p>
                <p className="text-xs text-text-muted mt-1">Drag and drop or click to browse</p>
              </>
            )}
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
                  <p className="text-[10px] text-gray-500 mt-1">Binary Mask ({pipeline === 'morphological' ? 'Morph' : 'Directional'})</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visualizer & Results */}
        <div className="lg:col-span-2 space-y-6">
          
          {hasResult ? (
            <>
              {/* Main Visualizer */}
              <div className="card p-2 bg-white">
                <BeforeAfterSlider 
                  beforeImage={`${API_BASE_URL}${resultData.images.original}`} 
                  afterImage={`${API_BASE_URL}${resultData.images.overlay}`} 
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
                  isDefect ? "border-l-danger-500" : "border-l-success-500"
                )}>
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xs font-medium text-text-muted uppercase tracking-wide">Analysis Status</p>
                      <h3 className={cn(
                        "text-xl font-bold mt-1",
                        isDefect ? "text-danger-600" : "text-success-600"
                      )}>
                        {statusStr}
                      </h3>
                      {isDefect && (
                        <p className="text-sm font-medium text-gray-900 mt-1">
                          Type: <span className="text-danger-500 bg-danger-50 px-2 py-0.5 rounded ml-1">{resultData.defect_type}</span>
                        </p>
                      )}
                    </div>
                    {isDefect 
                      ? <AlertTriangle className="text-danger-500" size={28} />
                      : <CheckCircle2 className="text-success-500" size={28} />
                    }
                  </div>
                  
                  <div className="mt-5">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="font-medium text-gray-700">Confidence Score</span>
                      <span className="font-bold text-gray-900">{(resultData.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div 
                        className={cn(
                          "h-2 rounded-full",
                          resultData.confidence > 0.9 ? "bg-success-500" : "bg-warning-500"
                        )} 
                        style={{ width: `${resultData.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Metrics Card */}
                <div className="card p-5">
                  <p className="text-xs font-medium text-text-muted uppercase tracking-wide mb-4">Extracted Features</p>
                  
                  <div className="grid grid-cols-2 gap-4">
                    {pipeline === 'morphological' ? (
                      <>
                        <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                          <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Max Area</p>
                          <p className="text-lg font-bold text-gray-900">{resultData.features?.max_area?.toFixed(1) || 0} <span className="text-xs font-normal text-gray-500">px²</span></p>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                          <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Eccentricity</p>
                          <p className="text-lg font-bold text-gray-900">{resultData.features?.min_eccentricity?.toFixed(2) || 0}</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                          <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Horiz. Length</p>
                          <p className="text-lg font-bold text-gray-900">{resultData.features?.horiz_length || 0} <span className="text-xs font-normal text-gray-500">px</span></p>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                          <p className="text-[10px] text-gray-500 font-medium uppercase mb-1">Vert. Length</p>
                          <p className="text-lg font-bold text-gray-900">{resultData.features?.vert_length || 0} <span className="text-xs font-normal text-gray-500">px</span></p>
                        </div>
                      </>
                    )}
                    
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 col-span-2 flex justify-between items-center">
                      <div>
                        <p className="text-[10px] text-gray-500 font-medium uppercase mb-0.5">Pipeline (Model)</p>
                        <p className="text-sm font-semibold text-gray-800 capitalize">{pipeline} ({modelType.toUpperCase()})</p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] text-gray-500 font-medium uppercase mb-0.5">Time</p>
                        <p className="text-sm font-bold text-primary-600">{resultData.inference_time_ms}ms</p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </>
          ) : (
            <div className="card h-full flex flex-col items-center justify-center p-10 bg-gray-50/50 border-dashed border-2 border-gray-200">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-4 shadow-sm text-gray-400">
                <ImageIcon size={32} />
              </div>
              <h3 className="text-lg font-semibold text-gray-700">Waiting for image</h3>
              <p className="text-sm text-gray-500 mt-2 text-center max-w-sm">
                Upload a fabric image using the panel on the left to see the AI defect detection results here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
