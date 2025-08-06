import React from "react";
import { FileText } from "lucide-react";
export const Header = () => {
  <header className="bg-white shadow-sm">
    <div className="container mx-auto px-4 py-5">
      <div className="flex items-center justify-center space-x-3">
        <FileText className="h-8 w-8 text-slate-600" />
        <h1 className="text-3xl font-bold text-slate-800 tracking-tight">AI Document Analyzer</h1>
      </div>
    </div>
  </header>
}