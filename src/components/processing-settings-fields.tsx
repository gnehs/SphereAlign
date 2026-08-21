import { i18n, type MessageDescriptor } from "@lingui/core";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { AlertTriangle, X } from "lucide-react";
import type { Dispatch, SetStateAction } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
  FieldTitle,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import {
  candidateMultiplierFor,
  customLutPathIsInvalid,
  featureModelPaths,
  gpuDeviceLabel,
  MASK_CLASSES,
  MASK_CLASS_LABELS,
  MAX_CANDIDATE_MULTIPLIER,
  MIN_CANDIDATE_MULTIPLIER,
  type ColorInspectionSummary,
  type DoctorReport,
  type PipelineSettings,
  type FeaturePipeline,
} from "@/lib/pipeline";

export interface ProcessingSettingsFieldsProps {
  settings: PipelineSettings;
  onSettingsChange: Dispatch<SetStateAction<PipelineSettings>>;
  doctor: DoctorReport;
  onChooseLut: () => void | Promise<void>;
  onChooseFeatureModelDirectory: () => void | Promise<void>;
  onGpuPreferenceTouched: () => void;
  sourceColorInspection?: ColorInspectionSummary | null;
}

function translate(descriptor: MessageDescriptor) {
  return i18n._(descriptor);
}

export function ProcessingSettingsFields({
  settings,
  onSettingsChange,
  doctor,
  onChooseLut,
  onChooseFeatureModelDirectory,
  onGpuPreferenceTouched,
  sourceColorInspection,
}: ProcessingSettingsFieldsProps) {
  const candidateMultiplier = candidateMultiplierFor(settings.extract);
  const candidateFps = settings.extract.baseFps * candidateMultiplier;
  const colorMode = settings.extract.colorMode;
  const detectedLog = sourceColorInspection?.shouldApply === true
    || sourceColorInspection?.files?.some((file) => file.shouldApply === true) === true;
  const restoreLog = colorMode === "logRec709" || colorMode === "dlogMRec709" || (colorMode === "auto" && detectedLog);
  const detectedSources = sourceColorInspection?.files?.filter((file) => file.cameraModel || file.detectedProfile || file.recommendedLut) ?? [];
  const lutPath = settings.extract.lutPath?.trim() ?? "";
  const lutPathInvalid = customLutPathIsInvalid(lutPath);
  const featurePipeline = settings.align.featurePipeline;
  const usingAliked = featurePipeline !== "sift";
  const featurePipelineItems: Array<{ value: FeaturePipeline; label: string }> = [
    { value: "sift", label: t`SIFT (fast default)` },
    { value: "aliked-n32-lightglue", label: t`ALIKED-N32 + LightGlue` },
    { value: "aliked-n16rot-lightglue", label: t`ALIKED-N16Rot + LightGlue` },
  ];

  return (
    <section className="min-h-0 overflow-hidden border-l max-[920px]:overflow-visible max-[920px]:border-t max-[920px]:border-l-0" aria-labelledby="task-processing-settings-title">
      <div className="scroll-fade-y scroll-fade-8 h-full overflow-y-auto overscroll-contain px-7 py-6 [scrollbar-gutter:stable] max-[920px]:h-auto max-[920px]:overflow-visible max-[920px]:p-5 max-[920px]:[--scroll-fade-mask:none]">
        <h2 id="task-processing-settings-title" className="mb-4 text-lg font-semibold text-foreground"><Trans>Processing settings</Trans></h2>
        <FieldGroup className="gap-4 [&>[data-slot=field]]:rounded-lg [&>[data-slot=field]]:border [&>[data-slot=field]]:bg-card [&>[data-slot=field]]:p-3">
          <Field aria-labelledby="frame-extraction-settings-title">
            <FieldTitle id="frame-extraction-settings-title"><Trans context="settings section" comment="Pipeline stage settings for extracting frames.">Frame extraction</Trans></FieldTitle>
            <FieldContent>
              <Field className="min-h-7 border-0 bg-transparent px-0 py-0.5">
                <FieldLabel htmlFor="base-fps"><Trans comment="Base frames-per-second setting for source media extraction.">Base frame rate (FPS)</Trans></FieldLabel>
                <Input
                  id="base-fps"
                  type="number"
                  min={1}
                  max={30}
                  step={1}
                  value={settings.extract.baseFps}
                  onChange={(event) => {
                    const baseFps = Math.min(30, Math.max(1, Number(event.currentTarget.value) || 1));
                    onSettingsChange((current) => {
                      const multiplier = candidateMultiplierFor(current.extract);
                      return { ...current, extract: { ...current.extract, baseFps, denseFps: baseFps * multiplier } };
                    });
                  }}
                />
              </Field>
              <Field orientation="horizontal" className="mt-2.5 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="sharpness-filter"
                  checked={settings.extract.skipBlurry}
                  onCheckedChange={(checked) => onSettingsChange((current) => ({ ...current, extract: { ...current.extract, skipBlurry: checked === true } }))}
                />
                <FieldLabel htmlFor="sharpness-filter"><Trans context="frame extraction setting" comment="Whether blurry candidate frames should be filtered out.">Sharpness filtering</Trans></FieldLabel>
              </Field>
              {settings.extract.skipBlurry && (
                <Field className="mt-2 min-h-7 border-0 bg-transparent px-0 py-0.5">
                  <div className="flex items-center justify-between">
                    <FieldTitle id="candidate-fps-label"><Trans comment="Frame rate used to sample candidate frames before selecting the sharpest ones.">Candidate frame rate</Trans></FieldTitle>
                    <span className="ml-auto font-mono text-sm text-muted-foreground">{candidateMultiplier}× · {candidateFps} FPS</span>
                  </div>
                  <Slider
                    aria-labelledby="candidate-fps-label"
                    aria-valuetext={`${candidateMultiplier}× · ${candidateFps} FPS`}
                    min={MIN_CANDIDATE_MULTIPLIER}
                    max={MAX_CANDIDATE_MULTIPLIER}
                    step={1}
                    value={[candidateMultiplier]}
                    onValueChange={(value) => {
                      const multiplier = Array.isArray(value) ? value[0] : value;
                      if (multiplier === undefined) return;
                      onSettingsChange((current) => ({
                        ...current,
                        extract: { ...current.extract, denseFps: current.extract.baseFps * multiplier },
                      }));
                    }}
                  />
                  <div className="flex items-center justify-between font-mono text-sm text-muted-foreground" aria-hidden="true"><span>2×</span><span>10×</span></div>
                  <FieldDescription><Trans comment="Candidate frames are sampled at a multiple of the base frame rate, then sharper frames are selected.">Sample candidates at a multiple of the base frame rate, then select sharper frames.</Trans></FieldDescription>
                </Field>
              )}
            </FieldContent>
          </Field>
          <Field aria-labelledby="lut-settings-title">
            <FieldTitle id="lut-settings-title"><Trans context="settings section" comment="Lookup table settings for restoring source media color.">LUT settings</Trans></FieldTitle>
            <FieldContent>
              <Field orientation="horizontal" className="min-h-7 gap-2 border-0 bg-transparent px-0 py-0.5">
                <Switch
                  id="extract-color-mode"
                  checked={restoreLog}
                  onCheckedChange={(checked) => onSettingsChange((current) => ({
                    ...current,
                    extract: { ...current.extract, colorMode: checked ? "logRec709" : "native" },
                  }))}
                />
                <div>
                  <FieldLabel htmlFor="extract-color-mode"><Trans comment="Automatically select a verified camera-specific LUT for detected Log footage.">Restore Log color with the matching LUT</Trans></FieldLabel>
                  {colorMode === "auto" && detectedLog && (
                    <FieldDescription><Trans comment="Camera model and explicit media profile metadata selected a verified LUT.">Camera model and Log profile detected; the matching LUT is enabled.</Trans></FieldDescription>
                  )}
                </div>
              </Field>
              {detectedSources.map((source, index) => (
                <div key={`${source.cameraModel ?? "camera"}-${source.detectedProfile ?? index}`} className="rounded-md border bg-muted/30 px-3 py-2 text-sm">
                  <div className="font-medium">{source.cameraModel ?? t`Unknown camera`} · {source.detectedProfile ?? t`Unknown color profile`}</div>
                  {source.recommendedLut && (
                    <div className="mt-1 text-muted-foreground">
                      {source.recommendedLut.displayName} · <code>{source.recommendedLut.fileName}</code>{" "}
                      <a className="underline underline-offset-2" href={source.recommendedLut.sourceUrl} target="_blank" rel="noreferrer"><Trans>Official source</Trans></a>
                    </div>
                  )}
                </div>
              ))}
              <Field className="min-h-7 gap-2 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=alert]]:mt-0.5" data-invalid={lutPathInvalid || undefined}>
                <FieldLabel htmlFor="extract-lut-path"><Trans comment="Optional user-provided 3D LUT file path.">Custom LUT (optional)</Trans></FieldLabel>
                <div className="flex items-center gap-2 [&_input]:flex-1">
                  <Input
                    id="extract-lut-path"
                    value={lutPath}
                    placeholder={t`Leave blank to use the detected camera's official LUT`}
                    aria-invalid={lutPathInvalid || undefined}
                    onChange={(event) => onSettingsChange((current) => ({
                      ...current,
                      extract: { ...current.extract, lutPath: event.currentTarget.value || undefined },
                    }))}
                  />
                  <Button type="button" variant="outline" size="sm" onClick={() => void onChooseLut()}>{t`Choose .cube`}</Button>
                  {lutPath && <Button type="button" variant="ghost" size="icon-xs" aria-label={t`Clear custom LUT`} onClick={() => onSettingsChange((current) => ({ ...current, extract: { ...current.extract, lutPath: undefined } }))}><X /></Button>}
                </div>
                {lutPathInvalid
                  ? <Alert variant="destructive"><AlertTriangle /><AlertTitle>{t`Invalid LUT file format`}</AlertTitle><AlertDescription>{t`Choose a 3D LUT with the .cube extension.`}</AlertDescription></Alert>
                  : colorMode === "logRec709" || colorMode === "dlogMRec709"
                    ? <FieldDescription>{t`When no custom file is specified, the runtime uses the detected model's verified LUT; otherwise it uses this .cube file.`}</FieldDescription>
                    : <FieldDescription>{t`Specify a .cube file only when you need to override the official LUT.`}</FieldDescription>}
              </Field>
            </FieldContent>
          </Field>
          <Field aria-labelledby="masking-settings-title">
            <FieldTitle id="masking-settings-title"><Trans context="settings section" comment="Pipeline stage settings for creating masks.">Masking</Trans></FieldTitle>
            <FieldContent>
              <Field orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="mask-yolo"
                  checked={settings.mask.yoloEnabled}
                  onCheckedChange={(checked) => onSettingsChange((current) => ({
                    ...current,
                    mask: {
                      ...current.mask,
                      yoloEnabled: checked === true,
                      classes: checked === true && current.mask.classes.length === 0 ? [...MASK_CLASSES] : current.mask.classes,
                    },
                  }))}
                />
                <FieldLabel htmlFor="mask-yolo"><Trans comment="Enable object detection masks from YOLO11.">YOLO object filtering</Trans></FieldLabel>
              </Field>
              {settings.mask.yoloEnabled && (
                <FieldGroup className="gap-2 px-0 pt-1 pb-2">
                  <FieldSet className="gap-2">
                    <FieldLegend variant="label"><Trans comment="Select one or more object classes to exclude from reconstruction.">Objects to mask (multiple selection)</Trans></FieldLegend>
                    <FieldGroup data-slot="checkbox-group" className="grid grid-cols-2 gap-x-3 gap-y-2">
                      {MASK_CLASSES.map((maskClass) => {
                        const checkboxId = `mask-class-${maskClass}`;
                        return (
                          <Field key={maskClass} orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                            <Checkbox
                              id={checkboxId}
                              checked={settings.mask.classes.includes(maskClass)}
                              onCheckedChange={(checked) => onSettingsChange((current) => {
                                const classes = checked === true
                                  ? Array.from(new Set([...current.mask.classes, maskClass]))
                                  : current.mask.classes.filter((value) => value !== maskClass);
                                return { ...current, mask: { ...current.mask, classes, yoloEnabled: classes.length > 0 } };
                              })}
                            />
                            <FieldLabel htmlFor={checkboxId}>{translate(MASK_CLASS_LABELS[maskClass])}</FieldLabel>
                          </Field>
                        );
                      })}
                    </FieldGroup>
                  </FieldSet>
                </FieldGroup>
              )}
              <Field orientation="horizontal" className="mt-2 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="mask-sky"
                  checked={settings.mask.maskSky}
                  onCheckedChange={(checked) => onSettingsChange((current) => ({ ...current, mask: { ...current.mask, maskSky: checked === true } }))}
                />
                <FieldLabel htmlFor="mask-sky"><Trans comment="Enable SkySeg sky masks.">Sky filtering</Trans></FieldLabel>
              </Field>
              {settings.mask.maskSky && <FieldDescription>{t`Use SkySeg to generate sky masks.`}</FieldDescription>}
              {!settings.mask.yoloEnabled && !settings.mask.maskSky && (
                <FieldDescription>{t`Masking is disabled; alignment starts after frame extraction.`}</FieldDescription>
              )}
            </FieldContent>
          </Field>
          <Field aria-labelledby="alignment-settings-title">
            <FieldTitle id="alignment-settings-title"><Trans context="settings section" comment="Pipeline stage settings for aligning source images and camera rigs.">Alignment</Trans></FieldTitle>
            <FieldContent>
              <div className="flex flex-col gap-2">
                <Field className="min-h-7 border-0 bg-transparent px-0 py-0.5">
                  <FieldLabel htmlFor="feature-pipeline"><Trans comment="Select the local feature extractor and matcher used by COLMAP alignment.">Feature matching method</Trans></FieldLabel>
                  <Select
                    items={featurePipelineItems}
                    value={featurePipeline}
                    onValueChange={(value) => {
                      const nextPipeline = (value ?? "sift") as FeaturePipeline;
                      onSettingsChange((current) => ({
                        ...current,
                        align: {
                          ...current.align,
                          featurePipeline: nextPipeline,
                          ...featureModelPaths(nextPipeline, current.align.featureModelDir ?? ""),
                          ...(nextPipeline !== "sift" && doctor.gpuAvailable === true ? { useGpu: true } : {}),
                        },
                      }));
                    }}
                  >
                    <SelectTrigger id="feature-pipeline" className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {featurePipelineItems.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  {featurePipeline === "sift" ? (
                    <FieldDescription><Trans comment="Explain the default SIFT feature matching option.">Fastest and most mature option for typical scenes.</Trans></FieldDescription>
                  ) : (
                    <Alert>
                      <AlertTriangle />
                      <AlertTitle><Trans comment="Heading for the learned feature matching performance tradeoff.">Slower, with a higher matching rate</Trans></AlertTitle>
                      <AlertDescription>
                        {featurePipeline === "aliked-n32-lightglue"
                          ? <Trans comment="Explain the ALIKED-N32 and LightGlue option.">Usually improves matching and camera registration in low-texture or difficult scenes. N32 is the recommended ALIKED option, but alignment takes longer.</Trans>
                          : <Trans comment="Explain the rotation-aware ALIKED-N16Rot and LightGlue option.">Usually improves matching under large viewpoint or rotation changes, but alignment takes longer.</Trans>}
                      </AlertDescription>
                    </Alert>
                  )}
                </Field>
                {usingAliked && (
                  <Field className="min-h-7 gap-2 border-0 bg-transparent px-0 py-0.5">
                    <FieldLabel htmlFor="feature-model-directory"><Trans comment="Directory containing the ALIKED extractor and LightGlue ONNX model files.">ALIKED model folder</Trans></FieldLabel>
                    <div className="flex items-center gap-2 [&_input]:flex-1">
                      <Input
                        id="feature-model-directory"
                        value={settings.align.featureModelDir ?? ""}
                        placeholder={t`Folder containing the ALIKED and LightGlue ONNX files`}
                        onChange={(event) => {
                          const featureModelDir = event.currentTarget.value;
                          onSettingsChange((current) => ({
                            ...current,
                            align: {
                              ...current.align,
                              featureModelDir,
                              ...featureModelPaths(current.align.featurePipeline, featureModelDir),
                            },
                          }));
                        }}
                      />
                      <Button type="button" variant="outline" size="sm" onClick={() => void onChooseFeatureModelDirectory()}><Trans>Choose folder</Trans></Button>
                    </div>
                    <FieldDescription>
                      {featurePipeline === "aliked-n32-lightglue"
                        ? <Trans>Requires aliked-n32.onnx and aliked-lightglue.onnx. The verified accelerated path uses an NVIDIA CUDA GPU.</Trans>
                        : <Trans>Requires aliked-n16rot.onnx and aliked-lightglue.onnx. The verified accelerated path uses an NVIDIA CUDA GPU.</Trans>}
                    </FieldDescription>
                  </Field>
                )}
                <Field orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5">
                  <Switch
                    id="use-intra-source-loop-closure"
                    size="sm"
                    checked={settings.align.useIntraSourceLoopClosure}
                    onCheckedChange={(checked) => onSettingsChange((current) => ({
                      ...current,
                      align: { ...current.align, useIntraSourceLoopClosure: checked },
                    }))}
                  />
                  <FieldContent>
                    <FieldLabel htmlFor="use-intra-source-loop-closure"><Trans comment="Find long-distance revisits within one source video to help close a reconstruction loop.">Single-video loop closure</Trans></FieldLabel>
                    <FieldDescription><Trans comment="Explain when the optional single-video loop-closure setting is useful.">Turn this on if the video passes through the same place again.</Trans></FieldDescription>
                    {settings.align.useIntraSourceLoopClosure && (
                      <Alert>
                        <AlertTriangle />
                        <AlertTitle><Trans>Possible incorrect matches</Trans></AlertTitle>
                        <AlertDescription><Trans comment="Warn that visually repetitive scenes can cause a false loop closure.">Similar-looking corridors or objects may be mistaken for a revisit, creating incorrect matches.</Trans></AlertDescription>
                      </Alert>
                    )}
                  </FieldContent>
                </Field>
                <Field orientation="horizontal" className="mt-2.5 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal" data-disabled={doctor.gpuAvailable === false || undefined}>
                  <Switch
                    id="use-gpu"
                    size="sm"
                    disabled={doctor.gpuAvailable === false}
                    checked={settings.align.useGpu}
                    onCheckedChange={(checked) => {
                      onGpuPreferenceTouched();
                      onSettingsChange((current) => ({ ...current, align: { ...current.align, useGpu: checked } }));
                    }}
                  />
                  <FieldContent>
                    <FieldLabel htmlFor="use-gpu"><Trans comment="Use CUDA acceleration for the COLMAP alignment stage.">Use CUDA acceleration for alignment</Trans></FieldLabel>
                    <FieldDescription>{doctor.gpuAvailable === false ? t`No usable COLMAP CUDA acceleration was detected, so the CPU will be used.` : t`Enabled by default when a CUDA-capable NVIDIA GPU is detected; falls back to the CPU if execution fails.`}</FieldDescription>
                  </FieldContent>
                </Field>
                {doctor.gpuAvailable === true && doctor.gpuDevices.length > 1 && (
                  <Field data-disabled={!settings.align.useGpu || undefined}>
                    <FieldLabel htmlFor="gpu-index"><Trans comment="Select which detected GPU should run alignment.">Select GPU</Trans></FieldLabel>
                    <Select
                      items={doctor.gpuDevices.map((device) => ({ value: String(device.index), label: gpuDeviceLabel(device, doctor.gpuDevices) }))}
                      value={settings.align.gpuIndex}
                      onValueChange={(gpuIndex) => onSettingsChange((current) => ({ ...current, align: { ...current.align, gpuIndex: gpuIndex ?? String(doctor.gpuDevices[0].index) } }))}
                      disabled={!settings.align.useGpu}
                    >
                      <SelectTrigger id="gpu-index" className="w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {doctor.gpuDevices.map((device) => <SelectItem key={device.index} value={String(device.index)}>{gpuDeviceLabel(device, doctor.gpuDevices)}</SelectItem>)}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </Field>
                )}
              </div>
            </FieldContent>
          </Field>
        </FieldGroup>
      </div>
    </section>
  );
}
