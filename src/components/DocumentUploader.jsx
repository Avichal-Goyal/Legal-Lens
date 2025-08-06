import React from 'react';
import { UploadCloud, FileCheck2, X } from 'lucide-react';

export const DocumentUploader = ({ file, setFile, onAnalyze, isLoading }) => {
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  return (
    <div className="bg-white p-6 md:p-8 rounded-xl shadow-md border border-slate-200 space-y-6">
      <h2 className="text-xl font-semibold text-center text-slate-700">Upload Your Document</h2>
      
      <div className="flex items-center justify-center w-full">
        <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-48 border-2 border-slate-300 border-dashed rounded-lg cursor-pointer bg-slate-50 hover:bg-slate-100 transition-colors">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <UploadCloud className="w-10 h-10 mb-3 text-slate-500" />
            <p className="mb-2 text-sm text-slate-500">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-slate-500">TXT, PDF, or DOCX files</p>
          </div>
          <input id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} />
        </label>
      </div>

      {file && (
        <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg animate-fade-in">
          <div className="flex items-center space-x-3">
            <FileCheck2 className="h-5 w-5 text-green-600" />
            <span className="text-sm font-medium text-green-800">{file.name}</span>
          </div>
          <button onClick={() => setFile(null)} className="p-1 rounded-full hover:bg-green-200">
            <X className="h-4 w-4 text-green-700" />
          </button>
        </div>
      )}

      <div className="pt-4 border-t border-slate-200">
        <button
          onClick={onAnalyze}
          disabled={!file || isLoading}
          className="w-full flex items-center justify-center px-6 py-3 text-base font-medium text-white bg-slate-800 border border-transparent rounded-lg shadow-sm hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-500 disabled:bg-slate-400 disabled:cursor-not-allowed transition-all"
        >
          {isLoading ? 'Analyzing...' : 'Analyze Document'}
        </button>
      </div>
    </div>
  );
};