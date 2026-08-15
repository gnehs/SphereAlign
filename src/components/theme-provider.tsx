import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type Theme = "system" | "light" | "dark";

const DEFAULT_THEME: Theme = "system";
const DEFAULT_STORAGE_KEY = "gs360studio.theme";
const DARK_MODE_QUERY = "(prefers-color-scheme: dark)";

type ThemeContextValue = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

function isTheme(value: string | null): value is Theme {
  return value === "system" || value === "light" || value === "dark";
}

function readStoredTheme(storageKey: string, fallback: Theme): Theme {
  try {
    const storedTheme = window.localStorage.getItem(storageKey);
    return isTheme(storedTheme) ? storedTheme : fallback;
  } catch {
    return fallback;
  }
}

function applyTheme(theme: Theme, systemPrefersDark?: boolean) {
  const resolvedTheme = theme === "system"
    ? ((systemPrefersDark ?? window.matchMedia(DARK_MODE_QUERY).matches) ? "dark" : "light")
    : theme;
  const root = window.document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(resolvedTheme);
  root.style.colorScheme = resolvedTheme;
}

export function initializeTheme(storageKey = DEFAULT_STORAGE_KEY) {
  applyTheme(readStoredTheme(storageKey, DEFAULT_THEME));
}

export function ThemeProvider({
  children,
  defaultTheme = DEFAULT_THEME,
  storageKey = DEFAULT_STORAGE_KEY,
}: {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
}) {
  const [theme, setThemeState] = useState<Theme>(() => readStoredTheme(storageKey, defaultTheme));

  useEffect(() => {
    const mediaQuery = window.matchMedia(DARK_MODE_QUERY);
    const updateSystemTheme = () => applyTheme(theme, mediaQuery.matches);
    updateSystemTheme();

    if (theme !== "system") return undefined;
    mediaQuery.addEventListener("change", updateSystemTheme);
    return () => mediaQuery.removeEventListener("change", updateSystemTheme);
  }, [theme]);

  const setTheme = useCallback((nextTheme: Theme) => {
    try {
      window.localStorage.setItem(storageKey, nextTheme);
    } catch (error) {
      console.info("[GS360] theme preference", error);
    }
    applyTheme(nextTheme);
    setThemeState(nextTheme);
  }, [storageKey]);

  const value = useMemo(() => ({ theme, setTheme }), [setTheme, theme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within ThemeProvider");
  return context;
}
