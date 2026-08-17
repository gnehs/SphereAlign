import { i18n, setupI18n } from "@lingui/core";
import type { I18n, Messages } from "@lingui/core";

export const locales = {
  en: "English",
  "zh-CN": "简体中文",
  "zh-TW": "繁體中文",
  ja: "日本語",
} as const;

export const localeLabels = locales;
export const supportedLocales = Object.keys(locales) as Locale[];
export type Locale = keyof typeof locales;

export const defaultLocale: Locale = "en";
export const DEFAULT_LOCALE = defaultLocale;
export const localeStorageKey = "spherealign.locale";

type CatalogModule = {
  messages: Messages;
};

const catalogLoaders = import.meta.glob<CatalogModule>(
  "./locales/*/messages.po",
);

let activationRequest = 0;
let englishI18nPromise: Promise<I18n> | undefined;

function normalizeLocale(locale: string | null | undefined): Locale | undefined {
  if (!locale) {
    return undefined;
  }

  const normalized = locale.trim().replace(/_/g, "-").toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === "en" || normalized.startsWith("en-")) {
    return "en";
  }

  if (
    normalized === "zh" ||
    normalized === "zh-cn" ||
    normalized === "zh-sg" ||
    normalized === "zh-hans" ||
    normalized.startsWith("zh-hans-")
  ) {
    return "zh-CN";
  }

  if (
    normalized === "zh-tw" ||
    normalized === "zh-hk" ||
    normalized === "zh-mo" ||
    normalized === "zh-hant" ||
    normalized.startsWith("zh-hant-")
  ) {
    return "zh-TW";
  }

  if (normalized === "ja" || normalized.startsWith("ja-")) {
    return "ja";
  }

  return undefined;
}

function readStoredLocale(): Locale | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }

  try {
    return normalizeLocale(window.localStorage.getItem(localeStorageKey));
  } catch {
    // Access to localStorage can be denied by privacy settings or sandboxing.
    return undefined;
  }
}

function readNavigatorLocale(): Locale | undefined {
  if (typeof navigator === "undefined") {
    return undefined;
  }

  const candidates = [
    ...(Array.isArray(navigator.languages) ? navigator.languages : []),
    navigator.language,
  ];

  return candidates.map(normalizeLocale).find((locale): locale is Locale => Boolean(locale));
}

/** Return the persisted locale, then the best browser match, then English. */
export function getInitialLocale(): Locale {
  return readStoredLocale() ?? readNavigatorLocale() ?? defaultLocale;
}

/** Return the locale currently active in Lingui. */
export function getLocale(): Locale {
  return normalizeLocale(i18n.locale) ?? defaultLocale;
}

function persistLocale(locale: Locale): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(localeStorageKey, locale);
  } catch {
    // Persistence is optional; a blocked storage implementation must not break i18n.
  }
}

async function loadCatalog(locale: Locale): Promise<Messages> {
  const path = `./locales/${locale}/messages.po`;
  const loader = catalogLoaders[path];
  if (!loader) {
    throw new Error(`Missing Lingui catalog for locale: ${locale}`);
  }

  const catalog = await loader();
  return catalog.messages;
}

/**
 * Return an isolated English translator for locale-independent exported text.
 * It must never replace the app-wide i18n instance used by the visible UI.
 */
export function getEnglishI18n(): Promise<I18n> {
  englishI18nPromise ??= loadCatalog(defaultLocale).then((messages) => setupI18n({
    locale: defaultLocale,
    messages: { [defaultLocale]: messages },
  }));
  return englishI18nPromise;
}

/**
 * Load and activate a supported locale. Unknown locales and catalog failures
 * safely fall back to the English catalog.
 */
export async function activateLocale(candidate: string | null | undefined): Promise<Locale> {
  const requestedLocale = normalizeLocale(candidate) ?? defaultLocale;
  const requestId = ++activationRequest;

  let activeLocale = requestedLocale;
  let messages: Messages;

  try {
    messages = await loadCatalog(requestedLocale);
  } catch {
    activeLocale = defaultLocale;
    try {
      messages = await loadCatalog(defaultLocale);
    } catch {
      // Keep the app renderable even before catalogs have been generated.
      messages = {};
    }
  }

  // A newer request may have started while this catalog was loading.
  if (requestId !== activationRequest) {
    return getLocale();
  }

  i18n.load(activeLocale, messages);
  i18n.activate(activeLocale);
  persistLocale(activeLocale);
  return activeLocale;
}

// Name used in Lingui's dynamic-loading guide; keep it as a stable alias.
export const dynamicActivate = activateLocale;
export const setLocale = activateLocale;

export { i18n };
