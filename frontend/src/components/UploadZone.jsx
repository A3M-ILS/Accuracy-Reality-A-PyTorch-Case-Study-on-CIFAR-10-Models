"use client";
import React, { useCallback, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { motion } from 'framer-motion';

export default function UploadZone({ onImageUpload, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, [disabled]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [disabled, onImageUpload]);

  const handleChange = function(e) {
    if (disabled) return;
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) return;
    
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onImageUpload(file);
  };

  return (
    <div className="w-full max-w-2xl mx-auto mt-8 mb-12">
      <motion.div
        whileHover={{ scale: previewUrl || disabled ? 1 : 1.02 }}
        whileTap={{ scale: previewUrl || disabled ? 1 : 0.98 }}
        className={`relative overflow-hidden rounded-2xl border-2 border-dashed transition-colors duration-300 ease-in-out cursor-pointer glass-panel
          ${isDragging ? 'border-sky-400 bg-sky-500/10' : 'border-slate-600 hover:border-slate-400 bg-slate-800/40'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleChange} 
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        />
        
        <div className="p-10 text-center flex flex-col items-center justify-center min-h-[250px]">
          {previewUrl ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative w-full max-w-xs rounded-xl overflow-hidden shadow-2xl z-0 pointer-events-none"
            >
              <img src={previewUrl} alt="Preview" className="w-full h-auto object-cover rounded-xl" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end justify-center pb-4">
                <span className="text-sm font-medium text-white px-3 py-1 bg-black/40 backdrop-blur-md rounded-full">
                  Click or Drop to change
                </span>
              </div>
            </motion.div>
          ) : (
            <>
              <motion.div 
                animate={{ y: [0, -10, 0] }} 
                transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                className="bg-sky-500/20 p-4 rounded-full mb-6"
              >
                <UploadCloud className="w-12 h-12 text-sky-400" />
              </motion.div>
              <h3 className="text-xl font-semibold mb-2 text-slate-200">Drag & Drop your image</h3>
              <p className="text-slate-400 max-w-sm mx-auto text-sm">
                Support for JPG, PNG. Will be resized appropriately for ResNet inference.
              </p>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
