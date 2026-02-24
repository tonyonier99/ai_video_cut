import type { Cut, Asset, TrackConfig } from '../types';

function toRational(seconds: number, fps: number): string {
  const frames = Math.round(seconds * fps);
  return `${frames}/${fps}s`;
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function computeFrameDuration(fps: number): string {
  if (fps <= 0) return '100/3000s';
  const scaled = Math.round(fps * 100);
  return `100/${scaled}s`;
}

function formatName(fps: number, height: number): string {
  const fpsInt = Math.round(fps);
  return `FFVideoFormat${height}p${fpsInt}`;
}

export function generateFCPXML(
  cuts: Cut[],
  assets: Asset[],
  tracks: TrackConfig[],
  projectName: string,
  primaryFps?: number
): string {
  const fps = primaryFps || 30;
  const frameDuration = computeFrameDuration(fps);

  const primaryAsset = assets.find(a => a.type === 'video');
  const width = primaryAsset?.width || 1920;
  const height = primaryAsset?.height || 1080;

  const assetMap = new Map<string, Asset>();
  for (const asset of assets) {
    assetMap.set(asset.id, asset);
  }

  const usedAssetIds = new Set<string>();
  for (const cut of cuts) {
    if (cut.assetId) {
      usedAssetIds.add(cut.assetId);
    }
  }

  const videoTrack = tracks.find(t => t.type === 'video') || { id: 0, type: 'video' as const, name: 'V1', visible: true, locked: false };
  const audioTracks = tracks.filter(t => t.type === 'audio');
  const textTracks = tracks.filter(t => t.type === 'text');

  const videoCuts = [...cuts.filter(c => c.trackId === videoTrack.id)].sort((a, b) => a.start - b.start);
  const audioCuts = cuts.filter(c => audioTracks.some(t => t.id === c.trackId));
  const textCuts = cuts.filter(c => textTracks.some(t => t.id === c.trackId));

  const allCuts = [...videoCuts, ...audioCuts, ...textCuts];
  const totalEnd = allCuts.length > 0
    ? Math.max(...allCuts.map(c => c.end))
    : 0;

  let resourcesXml = '';
  resourcesXml += `        <format id="r1" name="${formatName(fps, height)}" frameDuration="${frameDuration}" width="${width}" height="${height}"/>\n`;

  for (const assetId of usedAssetIds) {
    const asset = assetMap.get(assetId);
    if (!asset) continue;
    const assetDur = asset.duration ? toRational(asset.duration, fps) : toRational(totalEnd, fps);
    const src = asset.originalPath
      ? `file://${asset.originalPath}`
      : asset.url || `file://localhost/path/to/${escapeXml(asset.name)}`;
    const hasVideo = asset.type === 'video' || asset.type === 'image' ? '1' : '0';
    const hasAudio = asset.type === 'video' || asset.type === 'audio' ? '1' : '0';
    resourcesXml += `        <asset id="a_${escapeXml(assetId)}" name="${escapeXml(asset.name)}" src="${escapeXml(src)}" duration="${assetDur}" hasVideo="${hasVideo}" hasAudio="${hasAudio}" format="r1"/>\n`;
  }

  if (usedAssetIds.size === 0 && assets.length > 0) {
    const fallback = assets[0];
    const fbDur = fallback.duration ? toRational(fallback.duration, fps) : toRational(totalEnd, fps);
    const fbSrc = fallback.originalPath
      ? `file://${fallback.originalPath}`
      : fallback.url || `file://localhost/path/to/${escapeXml(fallback.name)}`;
    resourcesXml += `        <asset id="a_${escapeXml(fallback.id)}" name="${escapeXml(fallback.name)}" src="${escapeXml(fbSrc)}" duration="${fbDur}" hasVideo="1" hasAudio="1" format="r1"/>\n`;
  }

  let spineXml = '';

  let prevEnd = 0;
  for (const cut of videoCuts) {
    if (cut.start > prevEnd + 0.001) {
      const gapDur = cut.start - prevEnd;
      spineXml += `                        <gap offset="${toRational(prevEnd, fps)}" duration="${toRational(gapDur, fps)}"/>\n`;
    }

    const clipDur = cut.end - cut.start;
    const sourceIn = cut.sourceStart ?? 0;
    const ref = cut.assetId ? `a_${escapeXml(cut.assetId)}` : (assets.length > 0 ? `a_${escapeXml(assets[0].id)}` : 'a_fallback');
    spineXml += `                        <asset-clip ref="${ref}" offset="${toRational(cut.start, fps)}" duration="${toRational(clipDur, fps)}" start="${toRational(sourceIn, fps)}" name="${escapeXml(cut.label || 'Clip')}"/>\n`;

    prevEnd = cut.end;
  }

  for (const cut of audioCuts) {
    const clipDur = cut.end - cut.start;
    const sourceIn = cut.sourceStart ?? 0;
    const ref = cut.assetId ? `a_${escapeXml(cut.assetId)}` : (assets.length > 0 ? `a_${escapeXml(assets[0].id)}` : 'a_fallback');
    spineXml += `                        <asset-clip ref="${ref}" lane="-1" offset="${toRational(cut.start, fps)}" duration="${toRational(clipDur, fps)}" start="${toRational(sourceIn, fps)}" name="${escapeXml(cut.label || 'Audio Clip')}"/>\n`;
  }

  for (const cut of textCuts) {
    const clipDur = cut.end - cut.start;
    spineXml += `                        <title lane="1" offset="${toRational(cut.start, fps)}" duration="${toRational(clipDur, fps)}" name="${escapeXml(cut.label || 'Title')}"/>\n`;
  }

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
${resourcesXml}    </resources>
    <library>
        <event name="Antigravity Cut Export">
            <project name="${escapeXml(projectName)}">
                <sequence format="r1" duration="${toRational(totalEnd, fps)}" tcStart="0s" tcFormat="NDF">
                    <spine>
${spineXml}                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>`;

  return xml;
}

export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
