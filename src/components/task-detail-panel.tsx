import { AnimatePresence, useReducedMotion } from "motion/react";
import * as m from "motion/react-m";
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from "react";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export const TASK_DETAIL_DRAWER_TRANSITION = {
  duration: 0.28,
  ease: [0.32, 0.72, 0, 1],
} as const;

const TASK_DETAIL_FADE_TRANSITION = {
  duration: 0.2,
  ease: [0.23, 1, 0.32, 1],
} as const;

const TASK_DETAIL_TAB_TRANSITION = {
  duration: 0.2,
  ease: [0.77, 0, 0.175, 1],
} as const;

const TASK_DETAIL_PRESS_TRANSITION = {
  duration: 0.12,
  ease: [0.23, 1, 0.32, 1],
} as const;

const panelVariants = {
  closed: { transform: "translateX(100%)", opacity: 1 },
  open: { transform: "translateX(0%)", opacity: 1 },
} as const;

const reducedPanelVariants = {
  closed: { opacity: 0 },
  open: { opacity: 1 },
} as const;

const tabVariants = {
  enter: (direction: number) => ({
    transform: `translateX(${direction * 8}px)`,
    opacity: 0,
  }),
  center: {
    transform: "translateX(0px)",
    opacity: 1,
  },
  exit: (direction: number) => ({
    transform: `translateX(${direction * -8}px)`,
    opacity: 0,
  }),
} as const;

export type TaskDetailTab = "summary" | "records";

export interface TaskDetailPanelProps {
  open: boolean;
  title: ReactNode;
  description: ReactNode;
  leading: ReactNode;
  activeTab: TaskDetailTab;
  onTabChange: (tab: TaskDetailTab) => void;
  summary: ReactNode;
  records: ReactNode;
  footer: ReactNode;
  onClose: () => void;
  restoreFocusRef?: RefObject<HTMLElement | null>;
  escapeBlocked?: boolean;
  onExitComplete?: () => void;
}

const tabs: readonly TaskDetailTab[] = ["summary", "records"];

export function TaskDetailPanel({
  open,
  title,
  description,
  leading,
  activeTab,
  onTabChange,
  summary,
  records,
  footer,
  onClose,
  restoreFocusRef,
  escapeBlocked = false,
  onExitComplete,
}: TaskDetailPanelProps) {
  const shouldReduceMotion = useReducedMotion();
  const panelRef = useRef<HTMLElement>(null);
  const tabButtonRefs = useRef<Partial<Record<TaskDetailTab, HTMLButtonElement | null>>>({});
  const previousTabRef = useRef<TaskDetailTab>(activeTab);
  const [tabDirection, setTabDirection] = useState(1);

  useEffect(() => {
    if (!open) return;
    const frameId = window.requestAnimationFrame(() => panelRef.current?.focus({ preventScroll: true }));
    return () => window.cancelAnimationFrame(frameId);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented || escapeBlocked) return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [escapeBlocked, onClose, open]);

  useEffect(() => {
    if (previousTabRef.current === activeTab) return;
    previousTabRef.current = activeTab;
    setTabDirection(activeTab === "records" ? 1 : -1);
  }, [activeTab]);

  const changeTab = (nextTab: TaskDetailTab) => {
    if (nextTab === activeTab) return;
    previousTabRef.current = nextTab;
    setTabDirection(nextTab === "records" ? 1 : -1);
    onTabChange(nextTab);
  };

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight" && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const nextTab = event.key === "Home"
      ? "summary"
      : event.key === "End"
        ? "records"
        : activeTab === "summary"
          ? "records"
          : "summary";
    changeTab(nextTab);
    tabButtonRefs.current[nextTab]?.focus();
  };

  const handleExitComplete = () => {
    const restoreTarget = restoreFocusRef?.current;
    if (restoreTarget?.isConnected) restoreTarget.focus({ preventScroll: true });
    onExitComplete?.();
  };

  const variants = shouldReduceMotion ? reducedPanelVariants : panelVariants;
  const transition = shouldReduceMotion ? TASK_DETAIL_FADE_TRANSITION : TASK_DETAIL_DRAWER_TRANSITION;

  return (
    <AnimatePresence initial={false} onExitComplete={handleExitComplete}>
      {open && (
        <m.aside
          ref={panelRef}
          key="task-detail"
          className="fixed top-13 right-0 bottom-0 z-50 flex w-[min(460px,100vw)] flex-col gap-0 overflow-hidden rounded-tl-xl border bg-card p-0 text-sm text-foreground shadow-lg outline-none will-change-transform max-[760px]:w-screen max-[760px]:rounded-none"
          role="dialog"
          aria-labelledby="task-detail-title"
          aria-describedby="task-detail-description"
          tabIndex={-1}
          initial="closed"
          animate="open"
          exit="closed"
          variants={variants}
          transition={transition}
        >
          <header className="relative border-b px-6 pt-6 pb-4">
            <div className="flex min-w-0 items-center gap-3.5 pr-7">
              {leading}
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <h2 id="task-detail-title" className="truncate text-base font-semibold text-foreground">{title}</h2>
                <p id="task-detail-description" className="truncate font-mono text-sm text-muted-foreground">{description}</p>
              </span>
            </div>
            <m.button
              type="button"
              className="absolute top-3 right-3 inline-grid size-8 place-items-center rounded-lg text-muted-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50"
              whileTap={{ transform: "scale(0.94)" }}
              transition={TASK_DETAIL_PRESS_TRANSITION}
              onClick={onClose}
              aria-label={t`Close`}
            >
              <X className="size-4" />
            </m.button>
          </header>

          <div className="flex min-h-0 flex-1 flex-col gap-0">
            <div className="mx-5 grid h-10 shrink-0 grid-cols-2 border-b" role="tablist">
              {tabs.map((tab) => {
                const active = activeTab === tab;
                return (
                  <m.button
                    key={tab}
                    ref={(element) => { tabButtonRefs.current[tab] = element; }}
                    id={`task-detail-tab-${tab}`}
                    type="button"
                    className={cn("relative inline-flex min-w-0 items-center justify-center px-1.5 text-sm font-medium text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50", active && "text-foreground")}
                    role="tab"
                    aria-selected={active}
                    aria-controls={`task-detail-panel-${tab}`}
                    tabIndex={active ? 0 : -1}
                    onClick={() => changeTab(tab)}
                    onKeyDown={handleTabKeyDown}
                    whileTap={{ transform: "scale(0.98)" }}
                    transition={TASK_DETAIL_PRESS_TRANSITION}
                  >
                    {tab === "summary" ? <Trans>Work summary</Trans> : <Trans>Processing records</Trans>}
                    {active && <m.span className="absolute inset-x-0 -bottom-px h-0.5 bg-foreground" layoutId="task-detail-tab-indicator" transition={TASK_DETAIL_TAB_TRANSITION} />}
                  </m.button>
                );
              })}
            </div>

            <div className="relative min-h-0 flex-1 overflow-hidden">
              <AnimatePresence initial={false} mode="popLayout" propagate custom={tabDirection}>
                {activeTab === "summary" ? (
                  <m.div
                    key="summary"
                    id="task-detail-panel-summary"
                    className="scroll-fade-y scroll-fade-8 absolute inset-0 overflow-y-auto px-6 pb-6"
                    role="tabpanel"
                    aria-labelledby="task-detail-tab-summary"
                    custom={tabDirection}
                    variants={tabVariants}
                    initial="enter"
                    animate="center"
                    exit="exit"
                    transition={TASK_DETAIL_TAB_TRANSITION}
                  >
                    {summary}
                  </m.div>
                ) : (
                  <m.div
                    key="records"
                    id="task-detail-panel-records"
                    className="scroll-fade-y scroll-fade-8 absolute inset-0 overflow-y-auto px-6 pb-6"
                    role="tabpanel"
                    aria-labelledby="task-detail-tab-records"
                    custom={tabDirection}
                    variants={tabVariants}
                    initial="enter"
                    animate="center"
                    exit="exit"
                    transition={TASK_DETAIL_TAB_TRANSITION}
                  >
                    {records}
                  </m.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <footer className="mt-auto flex flex-col gap-2 border-t px-6 pt-4 pb-5.5">{footer}</footer>
        </m.aside>
      )}
    </AnimatePresence>
  );
}
