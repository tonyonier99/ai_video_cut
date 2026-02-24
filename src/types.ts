export interface Cut {
  id: string;
  start: number;       // Timeline position start
  end: number;         // Timeline position end
  sourceStart: number; // Start in source video
  sourceEnd: number;   // End in source video
  label: string;
  trackId: number;
  assetId?: string;
  style?: {
    scale: number;
    x: number;
    y: number;
    rotation: number;
    opacity: number;
    mirror: boolean;
  };
}

export interface Asset {
  id: string;
  type: 'video' | 'image' | 'audio';
  name: string;
  url: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  originalPath?: string;
  file?: File;
}

export interface TrackConfig {
  id: number;
  type: 'video' | 'audio' | 'text';
  name: string;
  visible: boolean;
  locked: boolean;
}

export interface SubtitleConfig {
  fontFamily: string;
  fontSize: number;
  primaryColor: string;
  useGradient: boolean;
  gradientStops: Array<{ color: string; offset: number }>;
  gradientAngle: number;
  safeZoneMargin: number;
  useOutline: boolean;
  outlineColor: string;
  outlineWidth: number;
  useShadow: boolean;
  shadowColor: string;
  shadowBlur: number;
  shadowOffsetX: number;
  shadowOffsetY: number;
  useBackground: boolean;
  backgroundColor: string;
  backgroundOpacity: number;
  borderRadius: number;
  paddingX: number;
  paddingY: number;
  verticalOffset: number;
}

export interface DragState {
  isDragging: boolean;
  type: 'scrub' | 'move' | 'trim-start' | 'trim-end' | 'new-asset' | 'marquee' | 'blade' | 'pan' | null;
  targetId: string | null;
  startX: number;
  initialValue: number;
  initialCuts?: Cut[];
  newAssetId?: string;
  newAssetDuration?: number;
  ghostTime?: number;
  ghostTrackId?: number;
  initialScrollLeft?: number;
}

export type ActiveTool = 'select' | 'blade' | 'text' | 'hand';
export type LeftPanelTab = 'project' | 'controls' | 'effects' | 'roughcut' | 'highlights' | 'subtitles';
export type AppView = 'welcome' | 'editor';

export interface TrackTheme {
  primary: string;
  secondary: string;
  border: string;
}

export interface JobStatus {
  progress: number;
  message: string;
  step: string;
}
