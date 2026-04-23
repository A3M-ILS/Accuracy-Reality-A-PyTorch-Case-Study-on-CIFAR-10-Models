"use client";
import React from 'react';
import { motion } from 'framer-motion';

export default function ModelCard({ title, data, isWinner, delay = 0 }) {
  if (!data) return null;

  const topPred = data.topPrediction;
  const conf = (data.confidence * 100).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      className={`relative p-6 rounded-2xl glass-panel ${isWinner ? 'bg-indigo-900/30 border-indigo-500/50 shadow-[0_0_30px_rgba(99,102,241,0.2)]' : 'border-slate-700/50'}`}
    >
      {isWinner && (
        <motion.div 
          animate={{ opacity: [0.5, 1, 0.5] }} 
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute -inset-[1px] rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-500 opacity-20 blur-sm pointer-events-none"
        />
      )}
      
      <div className="relative z-10">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-400">
            {title}
          </h2>
          {isWinner && (
            <span className="px-3 py-1 text-xs font-bold uppercase tracking-wider bg-indigo-500/20 text-indigo-300 rounded-full border border-indigo-500/30">
              Highest Confidence
            </span>
          )}
        </div>

        <div className="mb-8">
          <p className="text-slate-400 text-sm mb-1 uppercase tracking-wider font-semibold">Prediction</p>
          <div className="text-4xl font-black text-white capitalize flex items-end gap-3">
            {topPred}
            <span className="text-xl font-medium text-slate-400 mb-1">{conf}%</span>
          </div>
        </div>

        <div className="space-y-3">
          <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-2">Class Distribution</p>
          {data.predictions.slice(0, 5).map((p, idx) => (
            <div key={p.className} className="group cursor-default">
              <div className="flex justify-between text-sm mb-1">
                <span className={`capitalize transition-colors ${idx === 0 ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>
                  {p.className}
                </span>
                <span className={`transition-colors ${idx === 0 ? (isWinner ? 'text-indigo-300 font-bold' : 'text-sky-300 font-bold') : 'text-slate-500'}`}>
                  {(p.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${p.probability * 100}%` }}
                  transition={{ duration: 1, delay: delay + (idx * 0.1), ease: "easeOut" }}
                  className={`h-full rounded-full ${idx === 0 ? (isWinner ? 'bg-indigo-500' : 'bg-sky-500') : 'bg-slate-600'}`}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
