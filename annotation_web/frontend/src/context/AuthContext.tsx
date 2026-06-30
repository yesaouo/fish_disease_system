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

export type Role = "guest" | "editor" | "expert";

type AuthContextValue = {
  name: string | null;
  token: string | null;
  role: Role | null; // null = not authenticated (show login)
  isAuthenticated: boolean;
  isExpert: boolean;
  canEdit: boolean; // editor or expert
  login: (name: string, apiKey: string) => Promise<LoginResponse>;
  continueAsGuest: () => void;
  logout: () => void;
};

const NAME_KEY = "annotatorName";
const TOKEN_KEY = "annotatorToken";
const ROLE_KEY = "annotatorRole";

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
  const [role, setRole] = useState<Role | null>(() => {
    const raw = localStorage.getItem(ROLE_KEY);
    if (raw === "guest" || raw === "editor" || raw === "expert") return raw;
    return null;
  });

  useEffect(() => {
    setAuthToken(token);
  }, [token]);

  const login = useCallback(async (displayName: string, apiKey: string) => {
    const response = await apiLogin(displayName, apiKey);
    const resolvedRole: Role = response.role === "editor" ? "editor" : "expert";
    localStorage.setItem(NAME_KEY, response.name);
    localStorage.setItem(TOKEN_KEY, response.token);
    localStorage.setItem(ROLE_KEY, resolvedRole);
    setName(response.name);
    setToken(response.token);
    setRole(resolvedRole);
    return response;
  }, []);

  const continueAsGuest = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(NAME_KEY);
    localStorage.setItem(ROLE_KEY, "guest");
    setToken(null);
    setName(null);
    setRole("guest");
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(NAME_KEY);
    localStorage.removeItem(ROLE_KEY);
    setToken(null);
    setName(null);
    setRole(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      name,
      token,
      role,
      isAuthenticated: role != null,
      isExpert: role === "expert",
      canEdit: role === "editor" || role === "expert",
      login,
      continueAsGuest,
      logout
    }),
    [name, token, role, login, continueAsGuest, logout]
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
