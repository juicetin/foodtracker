/**
 * Edit session manager -- command pattern undo/redo for entry editing.
 *
 * Each edit operation is wrapped in an EditCommand with execute() and undo().
 * Commands are stored in a stack; a pointer tracks the current position.
 * Supports undo, redo, reset (restore to original snapshot), and
 * provides canUndo/canRedo helpers.
 */

import {
  updateIngredientWeight,
  updateIngredientName,
  removeIngredient as removeIngredientDb,
  addIngredient as addIngredientDb,
  updateDishName as updateDishNameDb,
  recalculateEntryTotals,
  type IngredientUpdate,
} from './entryEditorService';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single ingredient snapshot for undo/redo state capture. */
export interface IngredientSnapshot {
  id: string;
  name: string;
  amountG: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sugar: number;
}

/** A single dish snapshot. */
export interface DishSnapshot {
  id: string;
  name: string;
  cuisine: string | null;
  portionScale: number;
  ingredients: IngredientSnapshot[];
}

/** Full entry state snapshot for reset. */
export interface EntrySnapshot {
  id: string;
  mealType: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes: string | null;
  createdAt: string;
  photoUri: string | null;
  photos: Array<{ uri: string }>;
  dishes: DishSnapshot[];
}

/** A reversible command. */
export interface EditCommand {
  execute(): void;
  undo(): void;
  description: string;
}

/** The edit session holding the command stack and pointer. */
export interface EditSession {
  commands: EditCommand[];
  /** Points to the last executed command index. -1 means no commands executed. */
  pointer: number;
  originalSnapshot: EntrySnapshot;
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

export function createEditSession(snapshot: EntrySnapshot): EditSession {
  return {
    commands: [],
    pointer: -1,
    originalSnapshot: JSON.parse(JSON.stringify(snapshot)),
  };
}

export function executeCommand(session: EditSession, command: EditCommand): EditSession {
  // Truncate any commands after pointer (discard redo history)
  const commands = session.commands.slice(0, session.pointer + 1);
  commands.push(command);
  command.execute();
  return {
    ...session,
    commands,
    pointer: commands.length - 1,
  };
}

export function undo(session: EditSession): EditSession {
  if (!canUndo(session)) return session;
  session.commands[session.pointer].undo();
  return {
    ...session,
    pointer: session.pointer - 1,
  };
}

export function redo(session: EditSession): EditSession {
  if (!canRedo(session)) return session;
  const newPointer = session.pointer + 1;
  session.commands[newPointer].execute();
  return {
    ...session,
    pointer: newPointer,
  };
}

export function canUndo(session: EditSession): boolean {
  return session.pointer >= 0;
}

export function canRedo(session: EditSession): boolean {
  return session.pointer < session.commands.length - 1;
}

// ---------------------------------------------------------------------------
// Reset -- restore DB state from original snapshot
// ---------------------------------------------------------------------------

export function reset(session: EditSession): void {
  const snap = session.originalSnapshot;

  // Rebuild ingredients from snapshot by removing all current and re-adding
  // For simplicity: undo all commands in reverse order
  for (let i = session.pointer; i >= 0; i--) {
    session.commands[i].undo();
  }
}

// ---------------------------------------------------------------------------
// Concrete command classes
// ---------------------------------------------------------------------------

export class ChangeWeightCommand implements EditCommand {
  description: string;

  constructor(
    private ingredientId: string,
    private entryId: string,
    private oldAmountG: number,
    private newAmountG: number,
  ) {
    this.description = `Change weight ${Math.round(oldAmountG)}g -> ${Math.round(newAmountG)}g`;
  }

  execute(): void {
    updateIngredientWeight(this.ingredientId, this.newAmountG);
    recalculateEntryTotals(this.entryId);
  }

  undo(): void {
    updateIngredientWeight(this.ingredientId, this.oldAmountG);
    recalculateEntryTotals(this.entryId);
  }
}

export class AddIngredientCommand implements EditCommand {
  description: string;
  private addedId: string | null = null;

  constructor(
    private data: IngredientUpdate,
  ) {
    this.description = `Add ingredient "${data.name}"`;
  }

  execute(): void {
    this.addedId = addIngredientDb(this.data);
    recalculateEntryTotals(this.data.entryId);
  }

  undo(): void {
    if (this.addedId) {
      removeIngredientDb(this.addedId);
      recalculateEntryTotals(this.data.entryId);
    }
  }
}

export class RemoveIngredientCommand implements EditCommand {
  description: string;

  constructor(
    private ingredientId: string,
    private savedData: IngredientUpdate,
  ) {
    this.description = `Remove ingredient "${savedData.name}"`;
  }

  execute(): void {
    removeIngredientDb(this.ingredientId);
    recalculateEntryTotals(this.savedData.entryId);
  }

  undo(): void {
    // Re-add with original data (generates new ID, but that's acceptable for undo)
    addIngredientDb(this.savedData);
    recalculateEntryTotals(this.savedData.entryId);
  }
}

export class RenameIngredientCommand implements EditCommand {
  description: string;

  constructor(
    private ingredientId: string,
    private oldName: string,
    private newName: string,
  ) {
    this.description = `Rename ingredient "${oldName}" -> "${newName}"`;
  }

  execute(): void {
    updateIngredientName(this.ingredientId, this.newName);
  }

  undo(): void {
    updateIngredientName(this.ingredientId, this.oldName);
  }
}

export class RenameDishCommand implements EditCommand {
  description: string;

  constructor(
    private dishId: string,
    private oldName: string,
    private newName: string,
  ) {
    this.description = `Rename dish "${oldName}" -> "${newName}"`;
  }

  execute(): void {
    updateDishNameDb(this.dishId, this.newName);
  }

  undo(): void {
    updateDishNameDb(this.dishId, this.oldName);
  }
}
