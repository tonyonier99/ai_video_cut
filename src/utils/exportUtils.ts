import type { Cut } from '../types';

export function generateFCPXML(
  cuts: Cut[],
  duration: number,
  fileName: string,
  originalVideoPath: string | null
): string {
  const fps = 30;

  let xml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
        <asset id="a1" name="${fileName}" src="file://${originalVideoPath || `localhost/path/to/${fileName}`}" duration="${Math.round(duration * fps)}/30s" hasVideo="1" hasAudio="1"/>
    </resources>
    <library>
        <event name="Antigravity Cut Event">
            <project name="Antigravity Project">
                <sequence format="r1" duration="${Math.round(cuts.reduce((a, b) => a + (b.end - b.start), 0) * fps)}/30s" tcStart="0s" tcFormat="NDF">
                    <spine>`;

  let offset = 0;
  cuts.forEach((cut, i) => {
    const dur = cut.end - cut.start;
    const durFrames = Math.round(dur * fps);
    const startFrames = Math.round(cut.start * fps);
    const offsetFrames = Math.round(offset * fps);

    xml += `
                        <video name="${cut.label || 'Clip ' + (i + 1)}" offset="${offsetFrames}/30s" ref="a1" duration="${durFrames}/30s" start="${startFrames}/30s"/>`;
    offset += dur;
  });

  xml += `
                    </spine>
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
