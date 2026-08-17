import { FolderOpen, Info, Minus, Plus, Settings2, Square, X } from "lucide-react";
import { t } from "@lingui/core/macro";
import * as m from "motion/react-m";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Trans } from "@lingui/react/macro";

const APP_NOTICE_EASE = [0.22, 1, 0.36, 1] as const;
const IS_TAURI_RUNTIME = typeof window !== "undefined"
  && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
const IS_WINDOWS_RUNTIME = IS_TAURI_RUNTIME
  && typeof navigator !== "undefined"
  && /Windows/i.test(navigator.userAgent);
const IS_MACOS_RUNTIME = IS_TAURI_RUNTIME
  && typeof navigator !== "undefined"
  && /Macintosh|Mac OS X/i.test(navigator.userAgent);

export interface AppNoticeProps {
  message: string;
  onClose: () => void;
  avoidBottomAction?: boolean;
}

export function AppNotice({ message, onClose, avoidBottomAction = false }: AppNoticeProps) {
  return (
    <m.div
      initial={{ y: 8, opacity: 0, scale: 0.98 }}
      animate={{ y: 0, opacity: 1, scale: 1 }}
      exit={{ y: 6, opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.18, ease: APP_NOTICE_EASE }}
      className={cn("fixed bottom-5 left-5 z-60 flex w-[min(360px,calc(100vw-40px))] items-start gap-2.5 rounded-xl border bg-popover/95 p-3 text-sm text-popover-foreground shadow-md backdrop-blur-md", avoidBottomAction && "max-[760px]:bottom-18")}
      role="status"
      aria-atomic="true"
    >
      <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
        <Info className="size-4" aria-hidden="true" />
      </span>
      <span className="min-w-0 flex-1 self-center leading-relaxed break-words">{message}</span>
      <Button className="-mt-1 -mr-1 shrink-0" variant="ghost" size="icon-xs" onClick={onClose} aria-label={t`Close notification`}>
        <X />
      </Button>
    </m.div>
  );
}

export function WindowsWindowControls() {
  if (!IS_WINDOWS_RUNTIME) return null;

  const appWindow = getCurrentWindow();
  const runWindowCommand = (command: () => Promise<void>) => {
    void command().catch((error) => console.error("[SphereAlign] Window control", error));
  };

  return (
    <div className="flex self-stretch" aria-label={t`Window controls`}>
      <Button
        className="h-full w-11 rounded-none"
        variant="ghost"
        aria-label={t`Minimize window`}
        title={t`Minimize window`}
        onClick={() => runWindowCommand(() => appWindow.minimize())}
      >
        <Minus />
      </Button>
      <Button
        className="h-full w-11 rounded-none"
        variant="ghost"
        aria-label={t`Maximize or restore window`}
        title={t`Maximize or restore window`}
        onClick={() => runWindowCommand(() => appWindow.toggleMaximize())}
      >
        <Square />
      </Button>
      <Button
        className="h-full w-11 rounded-none hover:bg-destructive hover:text-destructive-foreground"
        variant="ghost"
        aria-label={t`Close window`}
        title={t`Close window`}
        onClick={() => runWindowCommand(() => appWindow.close())}
      >
        <X />
      </Button>
    </div>
  );
}

export interface AppHeaderProps {
  onNewTask: () => void;
  onOpenProject: () => void | Promise<void>;
  onOpenSettings: () => void;
}

export function AppHeader({ onNewTask, onOpenProject, onOpenSettings }: AppHeaderProps) {
  return (
    <header className={cn(
      "sticky top-0 z-40 flex min-h-13 shrink-0 items-center border-b bg-background/95 px-4 py-2 backdrop-blur-sm select-none max-[760px]:px-3.5 max-[760px]:py-3",
      IS_MACOS_RUNTIME && "pl-[86px]",
      IS_WINDOWS_RUNTIME && "pr-0 py-0",
    )}>
      <div className="flex min-h-9 w-full shrink-0 items-center self-stretch" data-tauri-drag-region={IS_TAURI_RUNTIME ? "" : undefined}>
        <h1 className="shrink-0 text-base font-semibold tracking-tight text-foreground">SphereAlign</h1>
        <div className="ml-auto flex items-center gap-2">
          <Button size="sm" onClick={onNewTask}><Plus data-icon="inline-start" /><Trans context="task action" comment="Create a new reconstruction task.">New reconstruction task</Trans></Button>
          <Button size="sm" variant="outline" onClick={() => void onOpenProject()}><FolderOpen data-icon="inline-start" /><Trans context="project action" comment="Open an existing resumable project.">Open project</Trans></Button>
          <div className="mx-1 flex h-5">
            <Separator orientation="vertical" />
          </div>
          <Button size="sm" variant="outline" onClick={onOpenSettings}><Settings2 data-icon="inline-start" /><Trans>Settings</Trans></Button>
        </div>
        <WindowsWindowControls />
      </div>
    </header>
  );
}
