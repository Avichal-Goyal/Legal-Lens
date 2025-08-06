import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export const SignUpPage = () => {
  const [email, setEmail] = useState('');
  const { signup } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    signup(email);
    navigate('/');
  };

  return (
    <div className="flex flex-col items-center justify-center pt-16">
      <div className="w-full max-w-sm p-8 space-y-4">
        <div className="text-center">
            <h1 className="text-2xl font-bold text-zinc-900">Create an account</h1>
            <p className="text-zinc-500">Start analyzing documents with LegalLens</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-xs font-medium text-zinc-600 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 bg-white border border-zinc-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:border-transparent transition-shadow"
            />
          </div>
          <button
            type="submit"
            className="w-full px-4 py-2.5 font-semibold text-white bg-zinc-800 rounded-md hover:bg-zinc-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-500 transition-colors"
          >
            Sign Up
          </button>
        </form>
        <p className="text-xs text-center text-zinc-500">
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-zinc-700 hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
};