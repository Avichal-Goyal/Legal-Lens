import React from 'react';
import { BookText, AlertTriangle } from 'lucide-react';

export const ResultsDisplay = ({ results, onReset }) => {
  const { summary, clauses } = results;

  return (
    <div className="bg-white p-6 md:p-8 rounded-xl shadow-md border border-slate-200 space-y-8">
      <div className="space-y-3">
        <div className="flex items-center space-x-3">
          <BookText className="h-6 w-6 text-slate-600" />
          <h2 className="text-2xl font-bold text-slate-800">Analysis Summary</h2>
        </div>
        <p className="text-slate-600 leading-relaxed">{summary}</p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center space-x-3">
          <AlertTriangle className="h-6 w-6 text-slate-600" />
          <h2 className="text-2xl font-bold text-slate-800">Key Clauses Extracted </h2>
        </div>
        <div className="space-y-6">
          {Object.entries(clauses).map(([category, clauseList]) => (
            <div key={category}>
              <h3 className="text-lg font-semibold capitalize text-slate-700 mb-2 pb-2 border-b">{category}</h3>
              {clauseList.length > 0 ? (
                <ul className="space-y-3 list-none">
                  {clauseList.map((clause, index) => (
                    <li key={index} className="p-3 bg-slate-50 border-l-4 border-slate-300 rounded-r-md text-slate-700">
                      {clause}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-slate-500 italic">No {category} clauses were found.</p>
              )}
            </div>
          ))}
        </div>
      </div>
      
      <div className="pt-6 border-t border-slate-200">
         <button
          onClick={onReset}
          className="w-full flex items-center justify-center px-6 py-3 text-base font-medium text-white bg-slate-800 border border-transparent rounded-lg shadow-sm hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-500 transition-all"
        >
          Analyze Another Document
        </button>
      </div>
    </div>
  );
};