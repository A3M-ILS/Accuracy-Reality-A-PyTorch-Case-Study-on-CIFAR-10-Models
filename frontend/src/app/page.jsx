"use client";
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import UploadZone from '@/components/UploadZone';
import ModelCard from '@/components/ModelCard';
import DiffExplainer from '@/components/DiffExplainer';

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(`Error ${response.status}: ${detail?.detail || response.statusText || 'Unknown error'}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to get predictions. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const getWinner = () => {
    if (!results) return null;
    if (results.resnet101.confidence > results.resnet18.confidence) {
      return 'resnet101';
    } else if (results.resnet18.confidence > results.resnet101.confidence) {
      return 'resnet18';
    }
    return 'tie';
  };

  const winner = getWinner();

  return (
    <main className="min-h-[100dvh] relative overflow-x-hidden flex flex-col items-center py-16 px-4 sm:px-6 lg:px-8 z-0">
      {/* Background glow effects */}
      <div className="fixed top-0 left-1/4 w-[500px] h-[500px] bg-indigo-600/20 rounded-full blur-[120px] -z-10 animate-glow"></div>
      <div className="fixed bottom-0 right-1/4 w-[500px] h-[500px] bg-sky-600/20 rounded-full blur-[120px] -z-10 animate-glow" style={{ animationDelay: '2s' }}></div>

      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center max-w-3xl mb-8 mt-8"
      >
        <h1 className="text-5xl md:text-6xl font-black mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-sky-400 via-indigo-400 to-purple-400 drop-shadow-sm">
          CIFAR10 Model Battle
        </h1>
        <p className="text-xl text-slate-300 font-medium leading-relaxed">
          Upload an image to see how standard ResNet18 <span className="text-slate-500 text-sm">(32x32)</span> compares against deep ResNet101 <span className="text-slate-500 text-sm">(224x224)</span>.
        </p>
      </motion.div>

      <UploadZone onImageUpload={handleUpload} disabled={loading} />

      {loading && (
        <motion.div 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }} 
          exit={{ opacity: 0 }}
          className="my-12 flex flex-col items-center text-sky-400"
        >
          <div className="w-16 h-16 border-4 border-slate-700 border-t-sky-400 rounded-full animate-spin mb-6 shadow-[0_0_15px_rgba(56,189,248,0.5)]"></div>
          <p className="font-semibold text-lg animate-pulse tracking-wide text-sky-200">Running inference on both models...</p>
        </motion.div>
      )}

      {error && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }} 
          animate={{ opacity: 1, scale: 1 }} 
          className="my-8 p-5 bg-red-950/50 border border-red-500/50 rounded-2xl text-red-200 max-w-2xl w-full text-center shadow-[0_0_30px_rgba(239,68,68,0.2)] backdrop-blur-md font-medium"
        >
          {error}
        </motion.div>
      )}

      <AnimatePresence>
        {results && !loading && (
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-6xl flex flex-col gap-8 pb-16"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
              <ModelCard 
                title="ResNet18 (11.1M Params)" 
                data={results.resnet18} 
                isWinner={winner === 'resnet18' || winner === 'tie'}
                delay={0.1}
              />
              <ModelCard 
                title="ResNet101 (42.5M Params)" 
                data={results.resnet101} 
                isWinner={winner === 'resnet101' || winner === 'tie'}
                delay={0.3}
              />
            </div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              <DiffExplainer 
                resnet18={results.resnet18} 
                resnet101={results.resnet101} 
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
