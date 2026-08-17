import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Gauge,
  MonitorCog,
  RefreshCw,
} from "lucide-react";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { type Theme } from "@/components/theme-provider";
import { getLocale, setLocale } from "@/i18n";
import {
  LANGUAGE_OPTIONS,
  diagnosticItemLabel,
  diagnosticStatusLabel,
  formatDoctorCheckedAt,
  iconForDiagnostic,
  localiseUserMessage,
  type DiagnosticStatus,
  type DoctorReport,
} from "@/lib/pipeline";
import { cn } from "@/lib/utils";

export interface SettingsSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  theme: Theme;
  setTheme: (theme: Theme) => void;
  doctor: DoctorReport;
  doctorEssentialReady: boolean;
  performanceWarnings: string[];
  generalDoctorWarnings: string[];
  performanceFallback: string;
  performanceStatus: DiagnosticStatus;
  isWindowsPlatform: boolean;
  colmapPath: string;
  setColmapPath: (value: string) => void;
  doctorLoading: boolean;
  runDoctor: (customColmapPath: string) => void | Promise<void>;
  copyDoctorReport: () => void | Promise<void>;
  openColmapPicker: () => void | Promise<void>;
}

export function SettingsSheet({
  open,
  onOpenChange,
  theme,
  setTheme,
  doctor,
  doctorEssentialReady,
  performanceWarnings,
  generalDoctorWarnings,
  performanceFallback,
  performanceStatus,
  isWindowsPlatform,
  colmapPath,
  setColmapPath,
  doctorLoading,
  runDoctor,
  copyDoctorReport,
  openColmapPicker,
}: SettingsSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[min(420px,100vw)] gap-0 border-l bg-card p-0 data-[side=right]:sm:max-w-[420px]" side="right">
        <SheetHeader className="border-b px-6 pt-6 pb-4"><SheetTitle className="text-base font-semibold"><Trans>Settings</Trans></SheetTitle><SheetDescription className="mt-2 text-sm leading-relaxed"><Trans>Choose English, Simplified Chinese, Traditional Chinese, or Japanese.</Trans></SheetDescription></SheetHeader>
        <div className="scroll-fade-y scroll-fade-8 flex-1 overflow-y-auto px-6">
          <section className="border-b py-4 last:border-b-0">
            <FieldSet>
              <FieldLegend variant="label"><Trans context="language setting" comment="Select the language used by the interface.">Language</Trans></FieldLegend>
              <Select
                items={LANGUAGE_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
                value={getLocale()}
                onValueChange={(value) => { if (value) void setLocale(value); }}
              >
                <SelectTrigger className="w-full" aria-label={t`Interface language`}><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {LANGUAGE_OPTIONS.map((option) => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </FieldSet>
          </section>
          <section className="border-b py-4 last:border-b-0">
            <FieldSet className="gap-2.5">
              <FieldLegend variant="label"><Trans context="theme setting" comment="Select the visual theme of the interface.">Interface theme</Trans></FieldLegend>
              <FieldDescription><Trans>Choose light, dark, or follow the system appearance automatically.</Trans></FieldDescription>
              <ToggleGroup
                className="grid w-full grid-cols-3 [&_[data-slot=toggle-group-item]]:w-full [&_[data-slot=toggle-group-item]]:min-w-0"
                variant="outline"
                size="sm"
                spacing={0}
                value={[theme]}
                onValueChange={(values) => {
                  const nextTheme = values[0] as Theme | undefined;
                  if (nextTheme) setTheme(nextTheme);
                }}
                aria-label={t`Interface theme`}
              >
                <ToggleGroupItem value="system"><Trans>System</Trans></ToggleGroupItem>
                <ToggleGroupItem value="light"><Trans>Light</Trans></ToggleGroupItem>
                <ToggleGroupItem value="dark"><Trans>Dark</Trans></ToggleGroupItem>
              </ToggleGroup>
            </FieldSet>
          </section>
          <section className="border-b py-4 last:border-b-0">
            <div className="mb-3 flex flex-col items-stretch gap-3">
              <div className="flex min-w-0 flex-1 flex-col gap-1"><h2 className="text-base font-semibold text-foreground"><Trans>Runtime environment</Trans></h2><span className="text-sm text-muted-foreground">{t`Last checked: ${formatDoctorCheckedAt(doctor.checkedAt)}`}</span></div>
              <div className="grid w-full grid-cols-2 gap-2 [&_[data-slot=button]]:w-full [&_[data-slot=button]]:min-w-0" role="group" aria-label={t`Diagnostic actions`}>
                <Button type="button" variant="outline" size="sm" disabled={doctorLoading || doctor.checkedAt === "Not checked yet"} onClick={() => void copyDoctorReport()}><Copy data-icon="inline-start" /><Trans>Copy diagnostics</Trans></Button>
                <Button type="button" size="sm" className={doctorLoading ? "[&_svg]:animate-spin [&_svg]:[animation-duration:750ms]" : undefined} disabled={doctorLoading} onClick={() => void runDoctor(colmapPath)}><RefreshCw data-icon="inline-start" />{doctorLoading ? <Trans>Checking</Trans> : <Trans>Check again</Trans>}</Button>
              </div>
            </div>
            <div className="mb-4 flex flex-col gap-2.5">
              <Alert className={doctorEssentialReady ? "border-emerald-600/30 bg-emerald-500/10 text-emerald-700 [&_[data-slot=alert-description]]:text-muted-foreground dark:text-emerald-400" : undefined} variant={doctorEssentialReady ? "default" : "destructive"} role={doctorEssentialReady ? "status" : "alert"}>
                {doctorEssentialReady ? <CheckCircle2 /> : <AlertTriangle />}
                <AlertTitle>{doctorEssentialReady ? <Trans>All required capabilities are available</Trans> : <Trans>Required capabilities need attention</Trans>}</AlertTitle>
                <AlertDescription>{doctorEssentialReady ? <Trans>Basic reconstruction can run. CUDA and hardware acceleration are optional; processing still works without them but frame extraction, feature matching, and reconstruction will be slower.</Trans> : <Trans>Missing required tools will block some stages. Address the items marked “Needs attention” below first.</Trans>}</AlertDescription>
              </Alert>
              {performanceStatus !== "ready" && <Alert className={cn("[&_[data-slot=alert-description]]:leading-relaxed [&_[data-slot=alert-description]]:text-muted-foreground", performanceStatus === "warning" ? "border-amber-600/40 bg-amber-500/10 text-amber-700 dark:text-amber-400" : "bg-muted text-muted-foreground")} role={performanceStatus === "warning" ? "alert" : "status"}>
                <Gauge />
                <AlertTitle>{performanceStatus === "warning" ? <Trans>Performance will be affected</Trans> : <Trans>Acceleration capabilities not confirmed</Trans>}</AlertTitle>
                <AlertDescription>
                  {performanceWarnings.length > 0
                    ? performanceWarnings.map((warning) => <p key={warning}>{localiseUserMessage(warning)}</p>)
                    : performanceStatus === "warning"
                      ? <p>{localiseUserMessage(performanceFallback)}</p>
                      : <p><Trans>Run the environment check to confirm whether CUDA, hardware decoding, or the CPU will be used.</Trans></p>}
                  {performanceStatus === "warning" && <p><Trans>Stages that fall back to the CPU can still run, but processing may take significantly longer.</Trans></p>}
                </AlertDescription>
              </Alert>}
            </div>
            {isWindowsPlatform && <Field><FieldLabel htmlFor="colmap-path"><Trans comment="Path to the COLMAP executable used by the Windows runtime.">COLMAP executable</Trans></FieldLabel><FieldContent><div className="flex items-center gap-2 [&_input]:flex-1"><Input id="colmap-path" value={colmapPath} placeholder={t`Leave blank to detect from PATH`} onChange={(event) => setColmapPath(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openColmapPicker()}>{t`Change path`}</Button></div><FieldDescription><Trans>For the official Windows portable build, select COLMAP.bat in the root folder; you can also specify a self-built colmap.exe.</Trans></FieldDescription></FieldContent></Field>}
            <div className="flex items-start gap-2.5 py-2.5"><MonitorCog className="mt-0.5 size-4 shrink-0 text-primary" strokeWidth={1.75} /><span className="flex min-w-0 flex-1 flex-col gap-1"><strong className="text-base font-medium text-foreground">{localiseUserMessage(doctor.platform)}</strong><small className="text-sm leading-snug break-anywhere text-muted-foreground">{localiseUserMessage(doctor.summary)}</small></span></div>
            <div className="border-t">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <article className="flex min-w-0 items-start gap-2.5 border-t py-2.5 first:border-t-0" key={item.label}><Icon className={cn("mt-0.75 size-4 shrink-0 text-muted-foreground", item.status === "ready" && "text-emerald-600", item.status === "warning" && "text-destructive")} strokeWidth={1.75} /><div className="flex min-w-0 flex-1 flex-col gap-1"><div className="flex min-w-0 items-start gap-2"><span className="flex min-w-0 flex-1 flex-col gap-1"><small className="text-sm leading-snug break-anywhere text-muted-foreground">{diagnosticItemLabel(item.label)}</small><strong className="text-base font-medium text-foreground">{localiseUserMessage(item.value)}</strong></span><Badge className="shrink-0 text-sm" variant={item.status === "warning" ? "destructive" : "outline"}>{diagnosticStatusLabel(item.status)}</Badge></div><p className="mt-0.5 text-sm leading-relaxed break-anywhere text-muted-foreground">{localiseUserMessage(item.detail)}</p>{item.details && item.details.length > 0 && <Accordion className="mt-1"><AccordionItem value={`${item.label}-details`}><AccordionTrigger className="w-fit py-1 text-sm font-medium text-primary"><Trans>View details</Trans></AccordionTrigger><AccordionContent><ul className="mt-2 mb-0.5 flex list-none flex-col gap-1.5 text-sm leading-relaxed text-muted-foreground">{item.details.map((detail) => <li className="border-l-2 pl-3 break-anywhere" key={detail}>{localiseUserMessage(detail)}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}</div></article>; })}</div>
            {generalDoctorWarnings.length > 0 && <Alert variant="destructive"><AlertTriangle /><AlertTitle><Trans>Needs attention</Trans></AlertTitle><AlertDescription>{generalDoctorWarnings.map((warning) => <p key={warning}>{localiseUserMessage(warning)}</p>)}</AlertDescription></Alert>}
          </section>
        </div>
        <SheetFooter className="border-t px-6 pt-4 pb-5.5"><Button variant="outline" onClick={() => onOpenChange(false)}><Trans>Close</Trans></Button></SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
