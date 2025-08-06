import React, { useState } from 'react';
import { DocumentUploader } from '../components/DocumentUploader';
import { ResultsDisplay } from '../components/ResultsDisplay';
import { Loader } from '../components/Loader';
import { ErrorMessage } from '../components/ErrorMessage';

// Define the URL where your backend is running.
const BACKEND_URL = 'http://localhost:8000';

export const AnalyzePage = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    // Reset state for a new analysis
    setIsLoading(true);
    setError('');
    setResults(null);

    // Use FormData to prepare the file for sending
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Make the API call to your backend's /upload endpoint
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Handle errors from the backend
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);

    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err.message || 'An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleReset = () => {
      setFile(null);
      setResults(null);
      setError('');
      setIsLoading(false);
  }

  return (
    <main className="container mx-auto p-4 md:p-8">
      <div className="max-w-3xl mx-auto">
        {!results && (
          <DocumentUploader
            file={file}
            setFile={setFile}
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
          />
        )}
        {isLoading && <Loader />}
        {error && <ErrorMessage message={error} />}
        {results && <ResultsDisplay results={results} onReset={handleReset} />}
      </div>
    </main>
  );
};