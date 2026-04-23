"use client";

import { useState } from 'react';

// Mock user credentials
export const MOCK_CREDENTIALS = {
  username: 'admin',
  password: 'admin123'
};

export interface AuthState {
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

// Helper to safely access localStorage (SSR-safe)
const getAuthFromStorage = (): boolean => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('auth') === 'true';
  }
  return false;
};

export function useAuth(): AuthState {
  const [isAuthenticated, setIsAuthenticated] = useState(() => getAuthFromStorage());

  const login = async (username: string, password: string): Promise<boolean> => {
    if (username === MOCK_CREDENTIALS.username && password === MOCK_CREDENTIALS.password) {
      if (typeof window !== 'undefined') {
        localStorage.setItem('auth', 'true');
      }
      setIsAuthenticated(true);
      return true;
    }
    return false;
  };

  const logout = () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth');
    }
    setIsAuthenticated(false);
  };

  return {
    isAuthenticated,
    login,
    logout
  };
}
