import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useHistory } from '../hooks/useHistory'
import type { Cut } from '../types'

const makeCut = (id: string, start: number, end: number): Cut => ({
  id,
  start,
  end,
  sourceStart: start,
  sourceEnd: end,
  label: `Cut ${id}`,
  trackId: 0,
})

describe('useHistory', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const createHook = () => {
    let currentCuts: Cut[] = []
    const setCuts = (fn: Cut[] | ((prev: Cut[]) => Cut[])) => {
      if (typeof fn === 'function') {
        currentCuts = fn(currentCuts)
      } else {
        currentCuts = fn
      }
    }
    return renderHook(() => useHistory({
      cuts: currentCuts,
      setCuts: setCuts as React.Dispatch<React.SetStateAction<Cut[]>>,
    }))
  }

  it('starts with canUndo=false and canRedo=false', () => {
    const { result } = createHook()
    expect(result.current.canUndo).toBe(false)
    expect(result.current.canRedo).toBe(false)
    expect(result.current.historyLength).toBe(0)
  })

  it('initializeHistory sets the first history entry', () => {
    const { result } = createHook()
    const cuts = [makeCut('a', 0, 5)]

    act(() => {
      result.current.initializeHistory(cuts)
    })

    expect(result.current.historyLength).toBe(1)
    expect(result.current.historyIndex).toBe(0)
    expect(result.current.canUndo).toBe(false)
    expect(result.current.canRedo).toBe(false)
  })

  it('addToHistory grows the history', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]
    const cuts2 = [makeCut('a', 0, 5), makeCut('b', 5, 10)]

    act(() => {
      result.current.initializeHistory(cuts1)
    })
    act(() => {
      result.current.addToHistory(cuts2)
    })

    expect(result.current.historyLength).toBe(2)
    expect(result.current.historyIndex).toBe(1)
    expect(result.current.canUndo).toBe(true)
    expect(result.current.canRedo).toBe(false)
  })

  it('handleUndo moves back in history', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]
    const cuts2 = [makeCut('a', 0, 5), makeCut('b', 5, 10)]

    act(() => { result.current.initializeHistory(cuts1) })
    act(() => { result.current.addToHistory(cuts2) })
    act(() => {
      result.current.handleUndo()
      vi.runAllTimers()
    })

    expect(result.current.historyIndex).toBe(0)
    expect(result.current.canUndo).toBe(false)
    expect(result.current.canRedo).toBe(true)
  })

  it('handleRedo moves forward in history', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]
    const cuts2 = [makeCut('a', 0, 5), makeCut('b', 5, 10)]

    act(() => { result.current.initializeHistory(cuts1) })
    act(() => { result.current.addToHistory(cuts2) })
    act(() => {
      result.current.handleUndo()
      vi.runAllTimers()
    })
    act(() => {
      result.current.handleRedo()
      vi.runAllTimers()
    })

    expect(result.current.historyIndex).toBe(1)
    expect(result.current.canUndo).toBe(true)
    expect(result.current.canRedo).toBe(false)
  })

  it('adding after undo truncates future', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]
    const cuts2 = [makeCut('a', 0, 5), makeCut('b', 5, 10)]
    const cuts3 = [makeCut('c', 0, 3)]

    act(() => { result.current.initializeHistory(cuts1) })
    act(() => { result.current.addToHistory(cuts2) })
    act(() => {
      result.current.handleUndo()
      vi.runAllTimers()
    })
    act(() => { result.current.addToHistory(cuts3) })

    expect(result.current.historyLength).toBe(2) // cuts1 + cuts3 (cuts2 is truncated)
    expect(result.current.historyIndex).toBe(1)
    expect(result.current.canRedo).toBe(false)
  })

  it('does not undo past the beginning', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]

    act(() => { result.current.initializeHistory(cuts1) })
    act(() => {
      result.current.handleUndo()
      vi.runAllTimers()
    })

    expect(result.current.historyIndex).toBe(0)
    expect(result.current.canUndo).toBe(false)
  })

  it('does not redo past the end', () => {
    const { result } = createHook()
    const cuts1 = [makeCut('a', 0, 5)]

    act(() => { result.current.initializeHistory(cuts1) })
    act(() => {
      result.current.handleRedo()
      vi.runAllTimers()
    })

    expect(result.current.historyIndex).toBe(0)
    expect(result.current.canRedo).toBe(false)
  })
})
