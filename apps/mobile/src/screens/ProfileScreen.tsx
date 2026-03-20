/**
 * ProfileScreen — nutrition goals editor, preferences, AI model management.
 */

import React, { useState, useEffect } from 'react';
import {
  ActivityIndicator,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList, UxMode } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import {
  loadExportEntries,
  loadExportRecipes,
  loadExportFavourites,
  loadExportOFFCache,
  generateCsv,
  generateJson,
} from '../services/export/exportService';
import { performIncrementalBackup, performFullBackup, listBackups } from '../services/backup/backupService';
import { getJournalCount } from '../services/backup/changeJournal';
import { registerAutoBackup } from '../services/backup/backupScheduler';
import type { BackupMetadata } from '../services/backup/types';
import { useSyncStore } from '../store/useSyncStore';

export default function ProfileScreen() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { nutritionGoals, setNutritionGoals, region, units, setRegion, setUnits, uxMode, setUxMode } = usePreferencesStore();

  const [editingGoals, setEditingGoals] = useState(false);
  const [calGoal, setCalGoal] = useState(String(nutritionGoals.calories));
  const [proteinGoal, setProteinGoal] = useState(String(nutritionGoals.protein));
  const [carbsGoal, setCarbsGoal] = useState(String(nutritionGoals.carbs));
  const [fatGoal, setFatGoal] = useState(String(nutritionGoals.fat));

  function saveGoals() {
    const cal = parseInt(calGoal, 10);
    const p = parseInt(proteinGoal, 10);
    const c = parseInt(carbsGoal, 10);
    const f = parseInt(fatGoal, 10);

    if ([cal, p, c, f].some((v) => isNaN(v) || v <= 0)) {
      Alert.alert('Invalid', 'All goals must be positive numbers.');
      return;
    }

    setNutritionGoals({ calories: cal, protein: p, carbs: c, fat: f });
    setEditingGoals(false);
  }

  function cancelGoalEdit() {
    setCalGoal(String(nutritionGoals.calories));
    setProteinGoal(String(nutritionGoals.protein));
    setCarbsGoal(String(nutritionGoals.carbs));
    setFatGoal(String(nutritionGoals.fat));
    setEditingGoals(false);
  }

  const regionLabels: Record<string, string> = {
    AU: 'Australia', US: 'United States', CA: 'Canada',
    UK: 'United Kingdom', FR: 'France', global: 'Global',
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Profile</Text>

      {/* Goals */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>Daily Goals</Text>
          {!editingGoals ? (
            <Pressable onPress={() => setEditingGoals(true)}>
              <Text style={styles.editBtn}>Edit</Text>
            </Pressable>
          ) : (
            <View style={styles.editActions}>
              <Pressable onPress={cancelGoalEdit}>
                <Text style={styles.cancelBtn}>Cancel</Text>
              </Pressable>
              <Pressable onPress={saveGoals}>
                <Text style={styles.saveBtn}>Save</Text>
              </Pressable>
            </View>
          )}
        </View>

        <GoalRow
          label="Calories"
          unit="kcal"
          value={calGoal}
          editing={editingGoals}
          onChange={setCalGoal}
          color="#EF4444"
        />
        <GoalRow
          label="Protein"
          unit="g"
          value={proteinGoal}
          editing={editingGoals}
          onChange={setProteinGoal}
          color="#3B82F6"
        />
        <GoalRow
          label="Carbs"
          unit="g"
          value={carbsGoal}
          editing={editingGoals}
          onChange={setCarbsGoal}
          color="#D97706"
        />
        <GoalRow
          label="Fat"
          unit="g"
          value={fatGoal}
          editing={editingGoals}
          onChange={setFatGoal}
          color="#16A34A"
        />
      </View>

      {/* Preferences */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Preferences</Text>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Region</Text>
          <Text style={styles.rowValue}>{regionLabels[region] ?? region}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Units</Text>
          <Text style={styles.rowValue}>{units === 'metric' ? 'Metric' : 'Imperial'}</Text>
        </View>
      </View>

      {/* Logging Mode */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Logging Mode</Text>
        <Text style={{ fontSize: 13, color: '#9CA3AF', marginBottom: 12 }}>
          How recipes are logged to your diary
        </Text>
        <UxModeSelector current={uxMode} onSelect={setUxMode} />
      </View>

      {/* Recipes */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recipes</Text>
        <Pressable style={styles.row} onPress={() => rootNavigation.navigate('Recipes')}>
          <Text style={styles.rowLabel}>My Recipes</Text>
          <Text style={styles.rowChevron}>→</Text>
        </Pressable>
      </View>

      {/* Export Data */}
      <ExportCard />

      {/* Backups */}
      <BackupCard />

      {/* Google Drive Sync */}
      <SyncCard />

      {/* Gallery Scan */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Gallery Scan</Text>
        <Pressable style={styles.row} onPress={() => rootNavigation.navigate('GalleryScan')}>
          <Text style={styles.rowLabel}>Gallery Scan Settings</Text>
          <Text style={styles.rowChevron}>→</Text>
        </Pressable>
      </View>

      {/* AI Models */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>AI Models</Text>
        <Pressable style={styles.row} onPress={() => rootNavigation.navigate('GeminiNanoTest')}>
          <Text style={styles.rowLabel}>Gemini Nano Test</Text>
          <Text style={styles.rowChevron}>→</Text>
        </Pressable>
      </View>

      {/* About */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>About</Text>
        <Pressable
          style={styles.row}
          onPress={() => Linking.openURL('https://openfoodfacts.org')}
        >
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Food product data provided by Open Food Facts</Text>
            <Text style={{ fontSize: 12, color: '#9CA3AF', marginTop: 2 }}>
              Licensed under the Open Database License (ODbL)
            </Text>
          </View>
          <Ionicons name="open-outline" size={16} color="#9CA3AF" />
        </Pressable>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function ExportCard() {
  const [exporting, setExporting] = useState(false);

  async function handleExport(format: 'csv' | 'json') {
    setExporting(true);
    try {
      const entries = loadExportEntries();
      const recipes = loadExportRecipes();
      const favourites = loadExportFavourites();
      const offCache = loadExportOFFCache();

      const content = format === 'csv'
        ? generateCsv(entries, recipes, favourites, offCache)
        : generateJson(entries, recipes, favourites, offCache);

      const ext = format === 'csv' ? 'csv' : 'json';
      const date = new Date().toISOString().split('T')[0];
      const filename = `tastimate-export-${date}.${ext}`;
      const fileUri = `${FileSystem.documentDirectory}${filename}`;

      await FileSystem.writeAsStringAsync(fileUri, content, {
        encoding: FileSystem.EncodingType.UTF8,
      });

      const canShare = await Sharing.isAvailableAsync();
      if (canShare) {
        await Sharing.shareAsync(fileUri, {
          mimeType: format === 'csv' ? 'text/csv' : 'application/json',
          dialogTitle: `Export ${format.toUpperCase()}`,
        });
      } else {
        Alert.alert('Saved', `Exported to ${fileUri}`);
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to export data.');
    } finally {
      setExporting(false);
    }
  }

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Export Data</Text>
      {exporting ? (
        <ActivityIndicator size="small" color="#16A34A" style={{ paddingVertical: 16 }} />
      ) : (
        <>
          <Pressable style={styles.exportBtn} onPress={() => handleExport('csv')}>
            <Ionicons name="document-text-outline" size={18} color="#16A34A" />
            <Text style={styles.exportBtnText}>Export as CSV</Text>
          </Pressable>
          <Pressable style={styles.exportBtn} onPress={() => handleExport('json')}>
            <Ionicons name="code-slash-outline" size={18} color="#3B82F6" />
            <Text style={[styles.exportBtnText, { color: '#3B82F6' }]}>Export as JSON</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function BackupCard() {
  const [backing, setBacking] = useState(false);
  const [backups, setBackups] = useState<BackupMetadata[]>([]);
  const [pendingChanges, setPendingChanges] = useState(0);

  useEffect(() => {
    setBackups(listBackups());
    setPendingChanges(getJournalCount());
    registerAutoBackup();
  }, []);

  function refreshState() {
    setBackups(listBackups());
    setPendingChanges(getJournalCount());
  }

  async function handleIncremental() {
    setBacking(true);
    try {
      const result = await performIncrementalBackup();
      if (result) {
        Alert.alert('Backup saved', `${result.filename}\n${result.changeCount} changes backed up.`);
      } else {
        Alert.alert('No changes', 'No changes to back up.');
      }
      refreshState();
    } catch {
      Alert.alert('Error', 'Failed to create incremental backup.');
    } finally {
      setBacking(false);
    }
  }

  async function handleFull() {
    setBacking(true);
    try {
      const result = await performFullBackup();
      Alert.alert('Full backup saved', `${result.filename}\nSize: ${formatBytes(result.sizeBytes)}`);
      refreshState();
    } catch {
      Alert.alert('Error', 'Failed to create full backup.');
    } finally {
      setBacking(false);
    }
  }

  const lastBackup = backups.length > 0 ? relativeTime(backups[0]!.createdAt) : 'Never';

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Backups</Text>

      <View style={styles.row}>
        <Text style={styles.rowLabel}>Last backup</Text>
        <Text style={styles.rowValue}>{lastBackup}</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.rowLabel}>Pending changes</Text>
        <Text style={styles.rowValue}>{pendingChanges}</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.rowLabel}>Total backups</Text>
        <Text style={styles.rowValue}>{backups.length}</Text>
      </View>

      {backing ? (
        <ActivityIndicator size="small" color="#16A34A" style={{ paddingVertical: 16 }} />
      ) : (
        <>
          <Pressable style={styles.exportBtn} onPress={handleIncremental}>
            <Ionicons name="cloud-upload-outline" size={18} color="#16A34A" />
            <Text style={styles.exportBtnText}>Incremental Backup</Text>
          </Pressable>
          <Pressable style={styles.exportBtn} onPress={handleFull}>
            <Ionicons name="download-outline" size={18} color="#3B82F6" />
            <Text style={[styles.exportBtnText, { color: '#3B82F6' }]}>Full Backup</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

function SyncCard() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { signedIn, userEmail, lastSyncAt, syncStatus } = useSyncStore();

  function statusColor(): string {
    switch (syncStatus) {
      case 'syncing': return '#F59E0B';
      case 'error': return '#EF4444';
      case 'conflict': return '#F59E0B';
      default: return '#16A34A';
    }
  }

  return (
    <Pressable style={styles.card} onPress={() => rootNavigation.navigate('SyncSettings')}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>Google Drive Sync</Text>
        <Text style={styles.rowChevron}>&#x2192;</Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.rowLabel}>Account</Text>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <View style={[syncStyles.dot, { backgroundColor: signedIn ? '#16A34A' : '#D1D5DB' }]} />
          <Text style={styles.rowValue}>{signedIn ? (userEmail ?? 'Connected') : 'Not connected'}</Text>
        </View>
      </View>

      <View style={styles.row}>
        <Text style={styles.rowLabel}>Last synced</Text>
        <Text style={styles.rowValue}>{lastSyncAt ? relativeTime(lastSyncAt) : 'Never'}</Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.rowLabel}>Status</Text>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          {syncStatus === 'syncing' ? (
            <ActivityIndicator size="small" color="#F59E0B" />
          ) : (
            <View style={[syncStyles.dot, { backgroundColor: statusColor() }]} />
          )}
          <Text style={styles.rowValue}>
            {syncStatus === 'idle' ? 'Up to date' : syncStatus === 'syncing' ? 'Syncing...' : syncStatus === 'error' ? 'Error' : 'Conflicts'}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

const syncStyles = StyleSheet.create({
  dot: { width: 8, height: 8, borderRadius: 4 },
});

const UX_MODE_OPTIONS: { mode: UxMode; label: string; desc: string }[] = [
  { mode: 'zero-effort', label: 'Zero-effort', desc: 'Auto-log, review later' },
  { mode: 'confirm-only', label: 'Confirm', desc: 'Review before logging' },
  { mode: 'guided-edit', label: 'Guided', desc: 'Step-by-step editing' },
];

function UxModeSelector({ current, onSelect }: { current: UxMode; onSelect: (m: UxMode) => void }) {
  return (
    <View style={styles.uxModeContainer}>
      {UX_MODE_OPTIONS.map(({ mode, label, desc }) => {
        const selected = current === mode;
        return (
          <Pressable
            key={mode}
            style={[styles.uxModeOption, selected && styles.uxModeOptionSelected]}
            onPress={() => onSelect(mode)}
          >
            <View style={[styles.uxModeRadio, selected && styles.uxModeRadioSelected]}>
              {selected && <View style={styles.uxModeRadioDot} />}
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[styles.uxModeLabel, selected && styles.uxModeLabelSelected]}>{label}</Text>
              <Text style={styles.uxModeDesc}>{desc}</Text>
            </View>
          </Pressable>
        );
      })}
    </View>
  );
}

function GoalRow({ label, unit, value, editing, onChange, color }: {
  label: string; unit: string; value: string; editing: boolean;
  onChange: (v: string) => void; color: string;
}) {
  return (
    <View style={styles.goalRow}>
      <View style={styles.goalLabelRow}>
        <View style={[styles.goalDot, { backgroundColor: color }]} />
        <Text style={styles.goalLabel}>{label}</Text>
      </View>
      {editing ? (
        <View style={styles.goalInputRow}>
          <TextInput
            style={styles.goalInput}
            value={value}
            onChangeText={onChange}
            keyboardType="number-pad"
            selectTextOnFocus
          />
          <Text style={styles.goalUnit}>{unit}</Text>
        </View>
      ) : (
        <Text style={styles.goalValue}>{value} {unit}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingTop: 60, paddingHorizontal: 16 },
  title: { fontSize: 28, fontWeight: '800', color: '#111827', marginBottom: 20 },

  card: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12,
  },
  cardTitle: { fontSize: 18, fontWeight: '700', color: '#111827' },
  editBtn: { fontSize: 15, fontWeight: '600', color: '#3B82F6' },
  editActions: { flexDirection: 'row', gap: 16 },
  cancelBtn: { fontSize: 15, fontWeight: '500', color: '#6B7280' },
  saveBtn: { fontSize: 15, fontWeight: '700', color: '#16A34A' },

  goalRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  goalLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  goalDot: { width: 10, height: 10, borderRadius: 5 },
  goalLabel: { fontSize: 15, fontWeight: '500', color: '#374151' },
  goalValue: { fontSize: 15, fontWeight: '600', color: '#111827' },
  goalInputRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  goalInput: {
    backgroundColor: '#F3F4F6', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
    fontSize: 15, fontWeight: '600', color: '#111827', minWidth: 70, textAlign: 'right',
  },
  goalUnit: { fontSize: 13, color: '#6B7280' },

  row: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  rowLabel: { fontSize: 15, color: '#374151' },
  rowValue: { fontSize: 15, color: '#6B7280' },
  rowChevron: { fontSize: 16, color: '#D1D5DB' },
  exportBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  exportBtnText: { fontSize: 15, fontWeight: '500', color: '#16A34A' },

  // UX Mode selector
  uxModeContainer: { gap: 8 },
  uxModeOption: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    paddingVertical: 12, paddingHorizontal: 12, borderRadius: 12,
    borderWidth: 1, borderColor: '#E5E7EB', backgroundColor: '#FAFAFA',
  },
  uxModeOptionSelected: {
    borderColor: '#7C3AED', backgroundColor: '#F5F3FF',
  },
  uxModeRadio: {
    width: 20, height: 20, borderRadius: 10,
    borderWidth: 2, borderColor: '#D1D5DB',
    justifyContent: 'center', alignItems: 'center',
  },
  uxModeRadioSelected: { borderColor: '#7C3AED' },
  uxModeRadioDot: {
    width: 10, height: 10, borderRadius: 5, backgroundColor: '#7C3AED',
  },
  uxModeLabel: { fontSize: 15, fontWeight: '600', color: '#374151' },
  uxModeLabelSelected: { color: '#7C3AED' },
  uxModeDesc: { fontSize: 12, color: '#9CA3AF', marginTop: 1 },
});
