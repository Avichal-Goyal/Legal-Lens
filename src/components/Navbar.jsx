import React from 'react';
import { Link } from 'react-router-dom';

export const Navbar = () => {
  return (
    <header className="bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-zinc-200">
      <nav className="container mx-auto px-6 py-3 flex justify-between items-center">
        {/* App Title/Logo */}
        <Link to="/" className="text-xl font-bold text-zinc-900">
          LegalLens
        </Link>
      </nav>
    </header>
  );
};