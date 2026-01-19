import React from 'react';
import { useCurrentFrame, spring, useVideoConfig } from 'remotion';

export interface AdvancedSubtitleConfig {
    fontSize: number;
    fontFamily: string;
    fontWeight: number | string;
    fontStyle: string;

    // Text Color & Gradient
    textColor: string;
    isTextGradient: boolean;
    textGradientColors: string[];
    textGradientDirection: string;

    // Stroke / Outline
    outlineWidth: number;
    outlineColor: string;

    // Shadow
    shadowColor: string;
    shadowBlur: number;
    shadowOffsetX: number;
    shadowOffsetY: number;
    shadowOpacity: number;

    // Spacing
    letterSpacing: number;
    lineHeight: number;
    textTransform: 'none' | 'uppercase' | 'lowercase' | 'capitalize';
    textAlign: 'left' | 'center' | 'right';

    // Layout
    marginBottom: number;
    animation?: 'none' | 'pop' | 'fade' | 'slide-up';
    animationDuration?: number;
    animationSpring?: number; // Mass or Intensity

    // Background
    isUnknownBackground: boolean;
    backgroundColor: string;
    backgroundOpacity: number;
    backgroundPaddingX: number;
    backgroundPaddingY: number;
    backgroundBorderRadius: number;
    charsPerLine?: number;
}

// Helper to wrap text
const wrapText = (text: string, maxChars: number) => {
    if (!maxChars || maxChars <= 0) return text;

    // Simple wrapping: insert \n every maxChars
    // Ideally we should respect word boundaries for English, but for mixed CJK often strict char count is preferred by users 
    // or a simple heuristic. Let's use a simple accumulation.

    let result = '';
    let currentLineLen = 0;

    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        // Treating CJK as 1 char count as per common subtitle editors, or could be 2.
        // Let's stick to simple length for now as per "subtitleCharsPerLine" name suggestion

        // If newline exists, reset
        if (char === '\n') {
            result += char;
            currentLineLen = 0;
            continue;
        }

        if (currentLineLen >= maxChars) {
            result += '\n';
            currentLineLen = 0;
        }

        result += char;
        currentLineLen++;
    }
    return result;
};

interface SubtitleProps {
    text: string;
    config: AdvancedSubtitleConfig;
}

export const Subtitle: React.FC<SubtitleProps> = ({ text, config }) => {
    const frame = useCurrentFrame();
    const { fps } = useVideoConfig();

    // DEBUG: Verify config reception
    // console.log(`[Subtitle] Text: ${text.substring(0, 10)}..., Config:`, config);

    // 1. Animation Logic
    const animType = config.animation || 'pop';
    const animDuration = config.animationDuration || 15;
    const animSpringMass = config.animationSpring || 0.5;

    // Spring configs
    const popScale = spring({
        fps,
        frame,
        config: { damping: 12, stiffness: 100, mass: animSpringMass },
        durationInFrames: animDuration,
    });

    const fadeOpacity = spring({
        fps,
        frame,
        config: { damping: 20 },
        durationInFrames: animDuration
    });

    const slideOffset = spring({
        fps,
        frame,
        config: { damping: 15, stiffness: 90, mass: animSpringMass },
        durationInFrames: Math.round(animDuration * 1.3) // Slide usually looks better slightly slower
    });

    // Determine final styles
    let transform = 'none';
    let opacity = 1;

    if (animType === 'pop') {
        transform = `scale(${popScale})`;
        opacity = fadeOpacity; // also slightly fade
    } else if (animType === 'fade') {
        opacity = fadeOpacity;
    } else if (animType === 'slide-up') {
        transform = `translateY(${(1 - slideOffset) * 50}px)`;
        opacity = fadeOpacity;
    } else if (animType === 'none') {
        // No animation
    }

    // ... Handle Gradient Text
    const colorStyle: React.CSSProperties = config.isTextGradient
        ? {
            backgroundImage: `linear-gradient(${config.textGradientDirection || 'to right'}, ${(config.textGradientColors || ['#fff', '#fff']).join(', ')})`,
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            color: 'transparent',
        }
        : { color: config.textColor || '#ffffff' };

    // ... Handle Shadow
    const shadowAlpha = Math.round((config.shadowOpacity ?? 0.8) * 255).toString(16).padStart(2, '0');
    const shadowColorHex = (config.shadowColor || '#000000').startsWith('#') ? config.shadowColor : '#000000';
    const finalShadowColor = `${shadowColorHex}${shadowAlpha}`;
    const textShadow = `${config.shadowOffsetX ?? 4}px ${config.shadowOffsetY ?? 4}px ${config.shadowBlur ?? 2}px ${finalShadowColor}`;

    // ... Handle Background
    const bgAlpha = Math.round((config.backgroundOpacity ?? 0.5) * 255).toString(16).padStart(2, '0');
    const bgColorHex = (config.backgroundColor || '#000000').startsWith('#') ? config.backgroundColor : '#000000';
    const finalBgColor = config.isUnknownBackground
        ? `${bgColorHex}${bgAlpha}`
        : 'transparent';

    const paddingX = config.backgroundPaddingX ?? 20;
    const paddingY = config.backgroundPaddingY ?? 10;
    const borderRadius = config.backgroundBorderRadius ?? 8;
    const fontSize = config.fontSize || 90;

    // Apply text wrapping
    const finalText = wrapText(text, config.charsPerLine ?? 0);

    const commonTextStyle: React.CSSProperties = {
        fontFamily: config.fontFamily ? `"${config.fontFamily}", Arial, sans-serif` : 'Arial',
        fontSize: `${fontSize}px`,
        fontWeight: config.fontWeight || 'normal',
        fontStyle: config.fontStyle || 'normal',
        letterSpacing: `${config.letterSpacing ?? 0}px`,
        lineHeight: config.lineHeight ?? 1.2,
        textTransform: (config.textTransform as any) || 'none',
        whiteSpace: 'pre-wrap',
        textAlign: (config.textAlign as any) || 'center',
        display: 'inline-block',
        gridArea: '1 / 1',
    };

    // Simplified rendering logic
    // If Monochrome: Use single <span> with paint-order (Native Browser Behavior - Best Quality)
    // If Gradient: Use Dual Layer (Back=White Fill+Stroke, Front=Gradient) to mimic paint-order masking

    if (!config.isTextGradient) {
        // MONOCHROME (Native Perfect)
        return (
            <div style={{
                // ...Container styles...
                width: '100%', height: '100%', display: 'flex', position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 100,
                justifyContent: config.textAlign === 'left' ? 'flex-start' : (config.textAlign === 'right' ? 'flex-end' : 'center'),
                alignItems: 'flex-end',
                paddingBottom: `${config.marginBottom ?? 150}px`,
                paddingLeft: '5%', paddingRight: '5%', boxSizing: 'border-box',
            }}>
                <div style={{
                    backgroundColor: finalBgColor,
                    padding: `${paddingY}px ${paddingX}px`,
                    borderRadius: `${borderRadius}px`,
                    transform: transform, opacity: opacity,
                    display: 'inline-flex', flexDirection: 'column',
                    alignItems: config.textAlign === 'left' ? 'flex-start' : (config.textAlign === 'right' ? 'flex-end' : 'center'),
                    maxWidth: '90%', justifyContent: 'center',
                }}>
                    <span style={{
                        ...commonTextStyle,
                        textShadow: textShadow,
                        WebkitTextStroke: config.outlineWidth > 0
                            ? `${(config.outlineWidth || 0) * 2}px ${config.outlineColor || '#000000'}`
                            : '0px transparent',
                        // Native masking magic
                        paintOrder: 'stroke fill',
                        strokeLinejoin: 'round',
                        strokeLinecap: 'round',
                        color: config.textColor || '#ffffff',
                    }}>
                        {finalText}
                    </span>
                </div>
            </div>
        );
    }

    // GRADIENT (Dual Layer Simulation)
    return (
        <div style={{
            width: '100%', height: '100%', display: 'flex', position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 100,
            justifyContent: config.textAlign === 'left' ? 'flex-start' : (config.textAlign === 'right' ? 'flex-end' : 'center'),
            alignItems: 'flex-end',
            paddingBottom: `${config.marginBottom ?? 150}px`,
            paddingLeft: '5%', paddingRight: '5%', boxSizing: 'border-box',
        }}>
            <div style={{
                backgroundColor: finalBgColor,
                padding: `${paddingY}px ${paddingX}px`,
                borderRadius: `${borderRadius}px`,
                transform: transform, opacity: opacity,
                display: 'inline-flex', flexDirection: 'column',
                alignItems: config.textAlign === 'left' ? 'flex-start' : (config.textAlign === 'right' ? 'flex-end' : 'center'),
                maxWidth: '90%', justifyContent: 'center',
            }}>
                <div style={{
                    display: 'inline-grid', position: 'relative',
                    textAlign: config.textAlign as any || 'center',
                    isolation: 'isolate',
                }}>
                    {/* Layer 1: Outline + Shadow (Bottom) */}
                    <span style={{
                        ...commonTextStyle,
                        textShadow: textShadow, // Shadow matches thick outline shape
                        WebkitTextStroke: config.outlineWidth > 0
                            ? `${(config.outlineWidth || 0) * 2}px ${config.outlineColor || '#000000'}`
                            : '0px transparent',
                        color: config.outlineColor || '#000000',
                        zIndex: 0,
                        userSelect: 'none',
                        gridArea: '1 / 1',
                        paintOrder: 'stroke fill',
                        strokeLinejoin: 'round',
                        strokeLinecap: 'round',
                        position: 'relative',
                    }}>
                        {finalText}
                    </span>

                    {/* Layer 2: Gradient Face (Top) */}
                    <span style={{
                        ...commonTextStyle,
                        WebkitTextStroke: '0px transparent',
                        textShadow: 'none',
                        zIndex: 10,
                        ...colorStyle,
                        gridArea: '1 / 1',
                        position: 'relative',
                    }}>
                        {finalText}
                    </span>
                </div>
            </div>
        </div>
    );
};
