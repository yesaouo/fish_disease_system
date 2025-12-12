import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import { login as apiLogin, setAuthToken } from "../api/client";
import type { LoginResponse } from "../api/types";

type AuthContextValue = {
  name: string | null;
  token: string | null;
  isAuthenticated: boolean;
  isExpert: boolean;
  login: (name: string, isExpert: boolean, apiKey: string) => Promise<LoginResponse>;
  logout: () => void;
};

const NAME_KEY = "annotatorName";
const TOKEN_KEY = "annotatorToken";
const EXPERT_KEY = "annotatorIsExpert";

const AuthContext = createContext<AuthContextValue | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children
}) => {
  const [name, setName] = useState<string | null>(() => {
    return localStorage.getItem(NAME_KEY);
  });
  const [token, setToken] = useState<string | null>(() => {
    return localStorage.getItem(TOKEN_KEY);
  });
  const [isExpert, setIsExpert] = useState<boolean>(() => {
    const raw = localStorage.getItem(EXPERT_KEY);
    if (raw == null) return true;
    return raw === "true";
  });

  useEffect(() => {
    setAuthToken(token);
  }, [token]);

  const login = useCallback(async (displayName: string, expert: boolean, apiKey: string) => {
    const response = await apiLogin(displayName, expert, apiKey);
    localStorage.setItem(NAME_KEY, response.name);
    localStorage.setItem(TOKEN_KEY, response.token);
    localStorage.setItem(EXPERT_KEY, String(expert));
    setName(response.name);
    setToken(response.token);
    setIsExpert(expert);
    return response;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(NAME_KEY);
    localStorage.removeItem(EXPERT_KEY);
    setToken(null);
    setName(null);
    setIsExpert(true);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      name,
      token,
      isAuthenticated: Boolean(name && token),
      isExpert,
      login,
      logout
    }),
    [name, token, isExpert, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
};
