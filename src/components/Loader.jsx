import React from 'react';

export const Loader = () => (
  <div className="flex flex-col items-center justify-center p-10 space-y-4">
    <div className="w-12 h-12 border-4 border-slate-300 border-t-slate-800 rounded-full animate-spin"></div>
    <p className="text-slate-600 font-medium">Processing document, please wait...</p>
  </div>
);