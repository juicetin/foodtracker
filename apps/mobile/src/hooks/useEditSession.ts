/**
 * useEditSession -- React hook wrapping editSessionManager for reactivity.
 *
 * Wraps the immutable session functions with useState so that
 * every command execution, undo, or redo triggers a re-render.
 */

import { useCallback, useRef, useState } from 'react';
import {
  createEditSession,
  executeCommand as execCmd,
  undo as undoCmd,
  redo as redoCmd,
  canUndo as canUndoFn,
  canRedo as canRedoFn,
  reset as resetFn,
  type EditSession,
  type EditCommand,
  type EntrySnapshot,
} from '../services/entryEditor/editSessionManager';

export interface UseEditSessionReturn {
  executeCommand: (command: EditCommand) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: () => void;
  commandCount: number;
  /** Initialize a new session from a snapshot. */
  initSession: (snapshot: EntrySnapshot) => void;
  /** Clear the session (e.g. after save). */
  clearSession: () => void;
}

export function useEditSession(): UseEditSessionReturn {
  const [session, setSession] = useState<EditSession | null>(null);
  // Counter to force re-renders on every mutation
  const [, setTick] = useState(0);
  const bump = useCallback(() => setTick((t) => t + 1), []);

  const initSession = useCallback((snapshot: EntrySnapshot) => {
    setSession(createEditSession(snapshot));
  }, []);

  const clearSession = useCallback(() => {
    setSession(null);
  }, []);

  const executeCommand = useCallback((command: EditCommand) => {
    setSession((prev) => {
      if (!prev) return prev;
      return execCmd(prev, command);
    });
    bump();
  }, [bump]);

  const undo = useCallback(() => {
    setSession((prev) => {
      if (!prev) return prev;
      return undoCmd(prev);
    });
    bump();
  }, [bump]);

  const redo = useCallback(() => {
    setSession((prev) => {
      if (!prev) return prev;
      return redoCmd(prev);
    });
    bump();
  }, [bump]);

  const reset = useCallback(() => {
    if (session) {
      resetFn(session);
      // Re-create session from original snapshot
      setSession(createEditSession(session.originalSnapshot));
    }
    bump();
  }, [session, bump]);

  return {
    executeCommand,
    undo,
    redo,
    canUndo: session ? canUndoFn(session) : false,
    canRedo: session ? canRedoFn(session) : false,
    reset,
    commandCount: session ? session.pointer + 1 : 0,
    initSession,
    clearSession,
  };
}
