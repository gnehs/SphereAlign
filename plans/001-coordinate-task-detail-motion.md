# 001 — Coordinate the task detail workspace motion

- **Status**: DONE
- **Commit**: e540f79
- **Severity**: HIGH
- **Category**: Cohesion & tokens / Interruptibility / Performance
- **Estimated scope**: 2 source files, roughly 250 lines moved or rewritten

## Problem

The task-detail interaction is controlled by four independent animation systems in `src/App.tsx`, so the interface starts at the same time but does not move as one system.

```tsx
// src/App.tsx:3099 — current main-canvas layout animation
<m.section ... layout layoutDependency={taskDetailOpen && Boolean(selectedTask)} transition={{ layout: TASK_DETAIL_TRANSITION }}>

// src/App.tsx:3252-3255 — current panel animation
initial={shouldReduceMotion ? { opacity: 0 } : { x: "100%", opacity: 1 }}
animate={{ x: 0, opacity: 1 }}
exit={shouldReduceMotion ? { opacity: 0 } : { x: "100%", opacity: 1 }}
transition={{ x: TASK_DETAIL_TRANSITION, opacity: TASK_DETAIL_OPACITY_TRANSITION }}

// src/App.tsx:3279 — current tab indicator animation
<m.span ... layoutId="task-detail-tab-indicator" transition={TASK_DETAIL_TRANSITION} />

// src/App.tsx:3284-3286 — current nested presence and tab content animation
<AnimatePresence initial={false} mode="sync">
  <m.div ... initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={TASK_DETAIL_TAB_TRANSITION}>
```

The nested `AnimatePresence` does not use `propagate`, so its active tab content does not receive an exit state when the outer task-detail panel exits. The panel also uses Motion's `x`/`y` shorthands instead of an explicit `transform`, while the main-canvas layout projection and tab indicator use separate transition semantics.

## Target

Create a dedicated `src/components/task-detail-panel.tsx` presentation component that owns the panel presence lifecycle, focus behavior, keyboard dismissal, ARIA tabs, tab direction, and all task-detail variants. `App.tsx` continues to own task data and business actions and passes the rendered summary, records, header-leading content, and footer actions as slots.

Use these exact tokens:

```ts
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
```

The normal-motion panel variants must use explicit transforms:

```ts
const panelVariants = {
  closed: { transform: "translateX(100%)", opacity: 1 },
  open: { transform: "translateX(0%)", opacity: 1 },
};
```

Reduced motion must remove translation and use `opacity: 0 -> 1` with the 0.2-second fade transition. Panel chrome may fade as one unit, but do not stagger the header, tabs, body, and footer; this is a crisp desktop workbench, not an onboarding flow.

Wrap the main task canvas and task-detail panel in `LayoutGroup id="task-detail-workspace"`. The main-canvas `layout` transition and panel transform must both use `TASK_DETAIL_DRAWER_TRANSITION`. Add `layout` to the main canvas's immediate content wrapper so Motion can correct scale distortion during the width projection.

The tab content may use directional transforms of `translateX(8px)` / `translateX(-8px)`, but both its opacity and transform must use `TASK_DETAIL_TAB_TRANSITION`. Use `AnimatePresence initial={false} mode="popLayout" propagate` and pass the tab direction through `custom` variants. The shared underline must use `TASK_DETAIL_TAB_TRANSITION`, not the drawer transition.

## Repo conventions to follow

- The app already uses `LazyMotion` and the strict `motion/react-m` entry point in `src/main.tsx:13-21`; continue importing DOM components as `* as m from "motion/react-m"`.
- The app already configures `MotionConfig reducedMotion="user"` in `src/main.tsx:18`; additionally use `useReducedMotion()` only where the target values must switch from translation to opacity.
- Use `cn()` from `@/lib/utils` for conditional classes and preserve semantic Tailwind tokens.
- Keep all Lingui `<Trans>` labels and existing task-domain output unchanged.

## Steps

1. Create `src/components/task-detail-panel.tsx` with a typed slot-based API. It must accept `open`, `title`, `description`, `leading`, `activeTab`, `onTabChange`, `summary`, `records`, `footer`, `onClose`, `restoreFocusRef`, `escapeBlocked`, and `onExitComplete`.
2. Move panel focus-on-open, Escape dismissal, focus restoration, ARIA tab keyboard navigation, and presence cleanup into the component. Escape must not close the panel while `escapeBlocked` is true.
3. Define the exact drawer, fade, and tab transitions above in the new component. Export only `TASK_DETAIL_DRAWER_TRANSITION` for the main canvas.
4. Replace independent panel props with `open` / `closed` variants. Use explicit `transform` strings and a reduced-motion opacity variant. Do not animate layout properties.
5. Make the panel chrome a single variant child. Add `propagate` to the tab content `AnimatePresence`; use directional custom variants and `mode="popLayout"`.
6. In `src/App.tsx`, remove task-detail motion tokens, panel refs, reduced-motion branching, Escape effect, and tab keyboard handler. Import the new component and shared drawer transition.
7. Keep the task-domain summary and records markup in `App.tsx` as slot content so the refactor does not export internal `Task`, `StageState`, or formatting helpers.
8. Wrap the non-empty task canvas and `TaskDetailPanel` in `LayoutGroup id="task-detail-workspace"`. Use the exported drawer transition for the canvas layout projection, and make its immediate grid wrapper a motion layout element with the same transition.
9. Preserve selected-task cleanup only after `onExitComplete`; do not clear it in the ordinary close path. Preserve deletion behavior for a task that is removed from the underlying collection.

## Boundaries

- Do NOT modify task processing, persistence, Tauri commands, translations, or task action behavior.
- Do NOT modify the settings Sheet or task creation Dialog.
- Do NOT add dependencies.
- Do NOT animate width, height, margin, padding, top, left, or right.
- Do NOT reintroduce Base UI Sheet or Tabs for task details.
- If the current code differs materially from the excerpts above, stop and report instead of improvising.

## Verification

- **Mechanical**: run `pnpm build` and `git diff --check`; both must pass.
- **Feel check**: run the Tauri app with at least one task and confirm:
  - the main task canvas and panel settle on the same frame during open and close;
  - panel header, tabs, body, and footer read as one rigid surface rather than four separate entrances;
  - rapidly close then reopen midway through exit and confirm the panel reverses from its current on-screen position without a jump;
  - switch summary/records repeatedly and confirm the content moves only 8px, the underline arrives on the same 200ms timeline, and no double-exposed scroll surface remains;
  - inspect at 10% playback speed and confirm main-canvas cards do not stretch while layout width changes;
  - enable reduced motion and confirm the panel cross-fades without positional movement.
- **Done when**: task-detail presentation and lifecycle live in one component, all participating motion uses the shared tokens above, and the mechanical checks pass.
