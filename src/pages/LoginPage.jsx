import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export const LoginPage = () => {
  const [email, setEmail] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    login(email);
    navigate('/');
  };

  return (
    <div className="flex flex-col items-center justify-center pt-16">
      <div className="w-full max-w-sm p-8 space-y-4">
        <div className="text-center">
            <h1 className="text-2xl font-bold text-zinc-900">Welcome back</h1>
            <p className="text-zinc-500">Log in to continue to LegalLens</p>
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
            Continue with Email
          </button>
        </form>
        <p className="text-xs text-center text-zinc-500">
          Don't have an account?{' '}
          <Link to="/signup" className="font-medium text-zinc-700 hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
};