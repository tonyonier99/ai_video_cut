import React from 'react';
import { Composition } from 'remotion';
import { MyComposition, myCompSchema } from './MyComposition';

export const RemotionRoot: React.FC = () => {
    return (
        <>
            <Composition
                id="MainComposition"
                component={MyComposition}
                durationInFrames={30 * 60 * 10} // Default duration (fallback)
                calculateMetadata={async ({ props }) => {
                    return {
                        durationInFrames: props.durationInFrames || 150,
                        props
                    };
                }}
                fps={30}
                width={1080}
                height={1920}
                schema={myCompSchema}
                defaultProps={{
                    videoUrl: '',
                    subtitles: [],
                    subtitleConfig: {
                        fontSize: 50,
                        fontFamily: 'Arial',
                        fontWeight: 400,
                        fontStyle: 'normal',

                        textColor: '#ffffff',
                        isTextGradient: false,
                        textGradientColors: ['#ffffff', '#ffffff'],
                        textGradientDirection: 'to right',

                        outlineWidth: 4,
                        outlineColor: '#000000',

                        shadowColor: '#000000',
                        shadowBlur: 0,
                        shadowOffsetX: 3,
                        shadowOffsetY: 3,
                        shadowOpacity: 0.8,

                        letterSpacing: 0,
                        lineHeight: 1.2,
                        textTransform: 'none',
                        textAlign: 'center',

                        marginBottom: 50,

                        isUnknownBackground: false,
                        backgroundColor: '#000000',
                        backgroundOpacity: 0.5,
                        backgroundPaddingX: 10,
                        backgroundPaddingY: 4,
                        backgroundBorderRadius: 4,
                    },
                    startFrom: 0,
                    duration: 0,
                    durationInFrames: 150
                }}
            />
        </>
    );
};
