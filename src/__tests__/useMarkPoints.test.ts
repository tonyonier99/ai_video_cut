import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useMarkPoints } from '../hooks/useMarkPoints';

describe('useMarkPoints', () => {
  const createOptions = (time: number = 5) => ({
    getCurrentTime: () => time,
    setCuts: vi.fn((updater: unknown) => {
      if (typeof updater === 'function') {
        return updater([]);
      }
      return updater;
    }),
    defaultTrackId: 0,
    defaultAssetId: 'a1',
    onHistoryCommit: vi.fn(),
  });

  it('marks in point at current time', () => {
    const opts = createOptions(3.5);
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markIn();
    });

    expect(result.current.inPoint).toBe(3.5);
    expect(result.current.outPoint).toBeNull();
  });

  it('marks out point at current time', () => {
    const opts = createOptions(8.0);
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markOut();
    });

    expect(result.current.outPoint).toBe(8.0);
    expect(result.current.inPoint).toBeNull();
  });

  it('clears outPoint if new inPoint is after it', () => {
    let time = 5;
    const opts = {
      ...createOptions(),
      getCurrentTime: () => time,
    };
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markOut();
    });
    expect(result.current.outPoint).toBe(5);

    time = 10;
    act(() => {
      result.current.markIn();
    });
    expect(result.current.inPoint).toBe(10);
    expect(result.current.outPoint).toBeNull();
  });

  it('clears inPoint if new outPoint is before it', () => {
    let time = 10;
    const opts = {
      ...createOptions(),
      getCurrentTime: () => time,
    };
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markIn();
    });
    expect(result.current.inPoint).toBe(10);

    time = 3;
    act(() => {
      result.current.markOut();
    });
    expect(result.current.outPoint).toBe(3);
    expect(result.current.inPoint).toBeNull();
  });

  it('clears both marks', () => {
    let time = 5;
    const opts = {
      ...createOptions(),
      getCurrentTime: () => time,
    };
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markIn();
    });
    time = 10;
    act(() => {
      result.current.markOut();
    });

    act(() => {
      result.current.clearMarks();
    });
    expect(result.current.inPoint).toBeNull();
    expect(result.current.outPoint).toBeNull();
  });

  it('createCutFromMarks does nothing without both points', () => {
    const opts = createOptions(5);
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markIn();
    });
    act(() => {
      result.current.createCutFromMarks();
    });

    expect(opts.setCuts).not.toHaveBeenCalled();
  });

  it('createCutFromMarks creates a cut and clears marks', () => {
    let time = 5;
    const setCutsMock = vi.fn((updater: unknown) => {
      if (typeof updater === 'function') {
        return updater([]);
      }
      return updater;
    });
    const opts = {
      getCurrentTime: () => time,
      setCuts: setCutsMock,
      defaultTrackId: 0,
      defaultAssetId: 'a1',
      onHistoryCommit: vi.fn(),
    };
    const { result } = renderHook(() => useMarkPoints(opts));

    act(() => {
      result.current.markIn();
    });
    time = 10;
    act(() => {
      result.current.markOut();
    });

    act(() => {
      result.current.createCutFromMarks();
    });

    expect(setCutsMock).toHaveBeenCalled();
    expect(result.current.inPoint).toBeNull();
    expect(result.current.outPoint).toBeNull();
  });
});
