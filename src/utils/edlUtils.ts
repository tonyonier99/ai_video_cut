import type { Cut, Asset } from '../types';

function pad(n: number): string {
  return n.toString().padStart(2, '0');
}

export function secondsToTimecode(seconds: number, fps: number): string {
  const totalSeconds = Math.max(0, seconds);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  const f = Math.floor((totalSeconds % 1) * fps);
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(f)}`;
}

function getReelName(asset: Asset | undefined, fallback: string): string {
  const name = asset?.name || fallback;
  const base = name.replace(/\.[^.]+$/, '');
  return base.substring(0, 8).toUpperCase().padEnd(8, ' ');
}

export function generateEDL(
  cuts: Cut[],
  assets: Asset[],
  projectName: string,
  fps: number
): string {
  const safeFps = fps > 0 ? fps : 30;
  const assetMap = new Map<string, Asset>();
  for (const asset of assets) {
    assetMap.set(asset.id, asset);
  }

  const sortedCuts = [...cuts].sort((a, b) => a.start - b.start);

  const lines: string[] = [
    `TITLE: ${projectName}`,
    'FCM: NON-DROP FRAME',
    '',
  ];

  let recOffset = 0;

  for (let i = 0; i < sortedCuts.length; i++) {
    const cut = sortedCuts[i];
    const eventNum = (i + 1).toString().padStart(3, '0');
    const asset = cut.assetId ? assetMap.get(cut.assetId) : assets[0];
    const reelName = getReelName(asset, `REEL${i + 1}`);

    const srcIn = cut.sourceStart ?? cut.start;
    const srcOut = cut.sourceEnd ?? cut.end;
    const clipDur = cut.end - cut.start;
    const recIn = recOffset;
    const recOut = recOffset + clipDur;

    const editType = asset?.type === 'audio' ? 'A' : 'V';

    lines.push(
      `${eventNum}  ${reelName}  ${editType}  C  ${secondsToTimecode(srcIn, safeFps)} ${secondsToTimecode(srcOut, safeFps)} ${secondsToTimecode(recIn, safeFps)} ${secondsToTimecode(recOut, safeFps)}`
    );
    lines.push(`* FROM CLIP NAME: ${cut.label || `Clip ${i + 1}`}`);

    recOffset = recOut;
  }

  lines.push('');
  return lines.join('\n');
}
