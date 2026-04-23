"use client";
import React from 'react';
import { motion } from 'framer-motion';

export default function DiffExplainer({ resnet18, resnet101 }) {
  if (!resnet18 || !resnet101) return null;

  const samePrediction = resnet18.topPrediction === resnet101.topPrediction;
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.6 }}
      className="mt-8 p-6 rounded-2xl glass-panel border-indigo-500/30 bg-gradient-to-br from-slate-900/90 to-indigo-900/30 shadow-2xl relative overflow-hidden"
    >
      <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>
      
      <div className="relative z-10">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-3">
          <span className="bg-indigo-500 w-1.5 h-6 rounded-full inline-block"></span>
          Why do they differ?
        </h3>
        
        <div className="text-slate-300 text-sm leading-relaxed space-y-4">
          <p className="bg-slate-950/40 p-4 rounded-xl border border-slate-700/50">
            <strong className="text-white">Observation:</strong> The models 
            {samePrediction 
              ? ` both strongly agree on the class "${resnet18.topPrediction}", but with differing confidence levels.` 
              : ` disagree! ResNet18 predicted "${resnet18.topPrediction}" while ResNet101 predicted "${resnet101.topPrediction}".`}
          </p>

          <p>
            <strong className="text-indigo-300">Architectural Depth:</strong> 
            ResNet18 has 18 layers and was trained natively on 32x32 images. It's fast and captures simpler hierarchical patterns. 
            ResNet101 is vastly deeper with 101 layers, trained on explicitly upscaled 224x224 images. This forces the deeper network to learn highly complex, abstract visual representations.
          </p>

          {!samePrediction && (
            <p>
              When they disagree, ResNet101's sheer depth gives it a much larger receptive field, allowing it to potentially catch subtle semantic textures that ResNet18 misses. However, the interpolation artifacts from upscaling a tiny 32x32 CIFAR image to 224x224 can occasionally mislead deeper networks.
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}
