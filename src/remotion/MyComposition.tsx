import React from 'react';
import { AbsoluteFill, Video, Sequence, staticFile, useCurrentFrame } from 'remotion';
import { Subtitle } from './Subtitle';
import { z } from 'zod';
import '../App.css'; // Import global styles and Google Fonts

// Helper component for dynamic speaker tracking
const DynamicVideo: React.FC<{
    src: string;
    startFromFrame: number;
    isFaceTracking: boolean;
    faceCenterX: number;
    speakerSegments: { start: number, end: number, faceCenterX: number }[];
    currentVideoTime: number; // Base time of this segment in original video
}> = ({ src, startFromFrame, isFaceTracking, faceCenterX, speakerSegments, currentVideoTime }) => {
    const FPS = 30;
    const frame = useCurrentFrame();

    // Calculate actual playback time in original video
    const actualVideoTime = currentVideoTime + (frame / FPS);

    // Determine current speaker center based on time
    let activeCenterX = faceCenterX;

    if (speakerSegments && speakerSegments.length > 0) {
        const activeSegment = speakerSegments.find(seg =>
            actualVideoTime >= seg.start && actualVideoTime < seg.end
        );
        if (activeSegment) {
            activeCenterX = activeSegment.faceCenterX;
        }
    }

    return (
        <Video
            src={src}
            startFrom={startFromFrame}
            muted={false}
            style={{
                height: '100%',
                width: '100%',
                objectFit: isFaceTracking ? 'cover' : 'contain',
                objectPosition: isFaceTracking ? `${activeCenterX * 100}% center` : 'center',
                // No transition - instant camera cut
            }}
        />
    );
};


export const myCompSchema = z.object({
    videoUrl: z.string(),
    audioUrl: z.string().optional(), // NEW: Processed Audio
    subtitles: z.array(z.object({
        id: z.string(),
        start: z.number(),
        end: z.number(),
        text: z.string(),
    })),
    subtitleConfig: z.object({
        fontSize: z.number(),
        fontFamily: z.string(),
        fontWeight: z.number().or(z.string()),
        fontStyle: z.string(),

        textColor: z.string(),
        isTextGradient: z.boolean(),
        textGradientColors: z.array(z.string()),
        textGradientDirection: z.string(),

        outlineWidth: z.number(),
        outlineColor: z.string(),

        shadowColor: z.string(),
        shadowBlur: z.number(),
        shadowOffsetX: z.number(),
        shadowOffsetY: z.number(),
        shadowOpacity: z.number(),

        letterSpacing: z.number(),
        lineHeight: z.number(),
        textTransform: z.string(),
        textAlign: z.string(),

        marginBottom: z.number(),
        animation: z.string().optional(),

        isUnknownBackground: z.boolean(),
        backgroundColor: z.string(),
        backgroundOpacity: z.number(),
        backgroundPaddingX: z.number(),
        backgroundPaddingY: z.number(),
        backgroundBorderRadius: z.number(),
        charsPerLine: z.number().optional()
    }),
    startFrom: z.number().optional(),
    duration: z.number().optional(),

    visualSegments: z.array(z.object({
        startInVideo: z.number(),
        duration: z.number(),
        zoom: z.number()
    })).optional(),
    isFaceTracking: z.boolean().optional(),
    faceCenterX: z.number().optional(),
    speakerSegments: z.array(z.object({
        start: z.number(),
        end: z.number(),
        faceCenterX: z.number()
    })).optional(),
    durationInFrames: z.number().optional()
});

export const MyComposition: React.FC<z.infer<typeof myCompSchema>> = ({
    videoUrl,
    subtitles,
    subtitleConfig,
    startFrom = 0,
    visualSegments,
    isFaceTracking = true,
    faceCenterX = 0.5,
    speakerSegments = []
}) => {
    const FPS = 30;
    // DEBUG: Check props received by Remotion
    console.log(`[MyComposition] FaceTracking=${isFaceTracking}, Center=${faceCenterX}, SpeakerSegs=${speakerSegments?.length || 0}`);
    console.log(`[MyComposition] SubtitleConfig:`, JSON.stringify(subtitleConfig, null, 2));

    // Inject Custom Font Style if needed
    // We assume backend is running on localhost:8000
    const customFontStyle = React.useMemo(() => {
        const fontName = subtitleConfig.fontFamily;
        // Skip common system fonts or fonts already in App.css (Inter, Noto Sans TC)
        if (['Arial', 'Inter', 'Noto Sans TC', 'Outfit', 'sans-serif'].includes(fontName)) {
            return null;
        }

        return (
            <style>{`
                @font-face {
                    font-family: '${fontName}';
                    src: url('http://localhost:8000/fonts/${fontName}.ttf') format('truetype'),
                         url('http://localhost:8000/fonts/${fontName}.otf') format('opentype');
                    font-weight: normal;
                    font-style: normal;
                }
            `}</style>
        );
    }, [subtitleConfig.fontFamily]);

    // Determine the actual video source
    // If it's a http/https/blob URL, use as-is
    // Otherwise, it's a filename in the public folder - use staticFile()
    const resolvedVideoUrl = React.useMemo(() => {
        if (!videoUrl) return '';
        if (videoUrl.startsWith('http') || videoUrl.startsWith('blob:')) {
            return videoUrl;
        }
        // It's a filename in public folder
        return staticFile(videoUrl);
    }, [videoUrl]);

    // Default to one segment if not provided (Legacy mode)
    const segments = visualSegments || [{
        startInVideo: startFrom,
        duration: 9999, // Will be clipped by durationInFrames of composition
        zoom: 1.0
    }];

    // Calculate layout for segments in the timeline
    let accumulatingFrame = 0;
    const layoutSegments = segments.map(seg => {
        const startFrame = accumulatingFrame;
        const durationFrames = Math.floor(seg.duration * FPS);
        accumulatingFrame += durationFrames;
        return { ...seg, startFrame, durationFrames };
    });

    return (
        <AbsoluteFill style={{ backgroundColor: '#000' }}>
            {customFontStyle}
            {/* Render Video Segments (Jump Cuts) with Dynamic Speaker Tracking */}
            {layoutSegments.map((seg, index) => (
                <Sequence
                    key={`seg-${index}`}
                    from={seg.startFrame}
                    durationInFrames={seg.durationFrames}
                >
                    {resolvedVideoUrl ? (
                        <div style={{
                            width: '100%', height: '100%',
                            transform: `scale(${seg.zoom})`,
                        }}>
                            <DynamicVideo
                                src={resolvedVideoUrl}
                                startFromFrame={Math.floor(seg.startInVideo * FPS)}
                                isFaceTracking={isFaceTracking}
                                faceCenterX={faceCenterX}
                                speakerSegments={speakerSegments}
                                currentVideoTime={seg.startInVideo}
                            />
                        </div>
                    ) : null}
                </Sequence>
            ))}

            {/* Master Audio Track - DISABLED to fix 'No Sound' issue. Using Video's internal audio instead.
            {audioUrl && (
                <Audio src={audioUrl} />
            )}
            */}

            {!videoUrl && (
                <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', color: 'gray', fontSize: 30 }}>
                    Waiting for Video...
                </AbsoluteFill>
            )}

            {/* Render Subtitles Re-synced to Segments */}
            {subtitles.map((sub) => {
                // Find which segment this subtitle belongs to (based on original time)
                // Note: A subtitle might span across a "gap". We clamp it to the segment.

                // We iterate all segments to see where this subtitle should appear in the NEW timeline
                return layoutSegments.map((seg, i) => {
                    // Check intersection in ORIGINAL time
                    const segStart = seg.startInVideo;
                    const segEnd = seg.startInVideo + seg.duration;

                    const subStart = sub.start;
                    const subEnd = sub.end;

                    const overlapStart = Math.max(segStart, subStart);
                    const overlapEnd = Math.min(segEnd, subEnd);

                    if (overlapEnd > overlapStart) {
                        // It overlaps! Calculate position in the NEW timeline (Sequence)
                        // Relative to segment start:
                        const offsetInSeg = overlapStart - segStart;


                        // Absolute frame in composition:
                        const showStartFrame = seg.startFrame + Math.floor(offsetInSeg * FPS);
                        const showDurationFrames = Math.max(1, Math.floor((overlapEnd - overlapStart) * FPS));

                        if (showDurationFrames > 0) {
                            return (
                                <Sequence
                                    key={`${sub.id}-${i}`}
                                    from={showStartFrame}
                                    durationInFrames={showDurationFrames}
                                >
                                    <Subtitle text={sub.text} config={{
                                        ...subtitleConfig,
                                        marginBottom: subtitleConfig.marginBottom // Explicit pass to ensure it's not lost
                                    } as any} />
                                </Sequence>
                            );
                        }
                    }
                    return null;
                });
            })}
        </AbsoluteFill>
    );
};
