import React from 'react';
import { AlertCircle } from 'lucide-react';

export const ErrorMessage = ({ message }) => (
  <div className="flex items-center p-4 space-x-3 bg-red-50 text-red-800 border border-red-200 rounded-lg animate-fade-in">
    <AlertCircle className="h-5 w-5" />
    <p className="text-sm font-medium">{message}</p>
  </div>
);