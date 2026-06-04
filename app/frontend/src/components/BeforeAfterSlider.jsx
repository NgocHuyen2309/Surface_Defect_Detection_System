import React, { useState, useRef } from 'react';

export default function BeforeAfterSlider({ beforeImage, afterImage }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);

  const handleMove = (e) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    let x = e.clientX - rect.left;
    if (e.touches && e.touches.length > 0) {
      x = e.touches[0].clientX - rect.left;
    }
    
    const position = Math.max(0, Math.min(x / rect.width * 100, 100));
    setSliderPosition(position);
  };

  const handleMouseDown = () => {
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseUp = () => {
    window.removeEventListener('mousemove', handleMove);
    window.removeEventListener('mouseup', handleMouseUp);
  };

  const handleTouchStart = () => {
    window.addEventListener('touchmove', handleMove);
    window.addEventListener('touchend', handleTouchEnd);
  };

  const handleTouchEnd = () => {
    window.removeEventListener('touchmove', handleMove);
    window.removeEventListener('touchend', handleTouchEnd);
  };

  return (
    <div 
      className="relative w-full aspect-video rounded-xl overflow-hidden select-none bg-gray-100 shadow-inner-soft"
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
    >
      {/* After Image (Background) */}
      <img src={afterImage} alt="Result Overlay" className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

      {/* Before Image (Foreground, clipped) */}
      <img 
        src={beforeImage} 
        alt="Original" 
        className="absolute inset-0 w-full h-full object-cover pointer-events-none" 
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }} 
      />

      {/* Slider Handle */}
      <div 
        className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize flex items-center justify-center z-10"
        style={{ left: `calc(${sliderPosition}% - 2px)` }}
      >
        <div className="w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center transform transition-transform hover:scale-110 pointer-events-none">
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" transform="rotate(90 12 12)" />
          </svg>
        </div>
      </div>
      
      {/* Labels */}
      <div className="absolute top-4 left-4 bg-gray-900/60 text-white text-xs px-2.5 py-1 rounded backdrop-blur-md pointer-events-none font-medium">
        Original Image
      </div>
      <div className="absolute top-4 right-4 bg-gray-900/60 text-white text-xs px-2.5 py-1 rounded backdrop-blur-md pointer-events-none font-medium">
        Result Overlay
      </div>
    </div>
  );
}
