/**
 * ProfileScreen — nutrition goals editor, preferences, AI model management.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  ActivityIndicator,
  Switch,
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
import type { RootStackParamList, UxMode, ThemePreference } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';
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
import {
  requestNotificationPermission,
  scheduleDailyNotification,
  cancelDailyNotification,
  buildMacroSummaryBody,
} from '../services/notifications/notificationService';
import {
  getContainers,
  deleteContainer,
  type Container,
} from '../services/scale/containerService';
import {
  isHealthConnectAvailable,
  initHealthConnect,
  requestWeightPermission,
} from '../services/health/healthConnectService';

const THEME_OPTIONS: { value: ThemePreference; label: string; icon: 'phone-portrait-outline' | 'sunny-outline' | 'moon-outline' }[] = [
  { value: 'system', label: 'System', icon: 'phone-portrait-outline' },
  { value: 'light', label: 'Light', icon: 'sunny-outline' },
  { value: 'dark', label: 'Dark', icon: 'moon-outline' },
];

export default function ProfileScreen() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { nutritionGoals, setNutritionGoals, region, units, setRegion, setUnits, uxMode, setUxMode } = usePreferencesStore();
  const themePreference = usePreferencesStore((s) => s.themePreference);
  const setThemePreference = usePreferencesStore((s) => s.setThemePreference);
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

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
    <ScrollView style={s.container} contentContainerStyle={s.content}>
      <Text style={s.title}>Profile</Text>

      {/* Goals */}
      <View style={s.card}>
        <View style={s.cardHeader}>
          <Text style={s.cardTitle}>Daily Goals</Text>
          {!editingGoals ? (
            <Pressable onPress={() => setEditingGoals(true)}>
              <Text style={s.editBtn}>Edit</Text>
            </Pressable>
          ) : (
            <View style={s.editActions}>
              <Pressable onPress={cancelGoalEdit}>
                <Text style={s.cancelBtn}>Cancel</Text>
              </Pressable>
              <Pressable onPress={saveGoals}>
                <Text style={s.saveBtn}>Save</Text>
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
          color={colors.accent.red}
          colors={colors}
        />
        <GoalRow
          label="Protein"
          unit="g"
          value={proteinGoal}
          editing={editingGoals}
          onChange={setProteinGoal}
          color={colors.accent.blue}
          colors={colors}
        />
        <GoalRow
          label="Carbs"
          unit="g"
          value={carbsGoal}
          editing={editingGoals}
          onChange={setCarbsGoal}
          color={colors.accent.amber}
          colors={colors}
        />
        <GoalRow
          label="Fat"
          unit="g"
          value={fatGoal}
          editing={editingGoals}
          onChange={setFatGoal}
          color={colors.accent.green}
          colors={colors}
        />
      </View>

      {/* Preferences */}
      <View style={s.card}>
        <Text style={s.cardTitle}>Preferences</Text>
        <Pressable
          style={s.row}
          onPress={() => {
            const regions = Object.keys(regionLabels);
            const currentIdx = regions.indexOf(region);
            const nextIdx = (currentIdx + 1) % regions.length;
            setRegion(regions[nextIdx] as any);
          }}
        >
          <Text style={s.rowLabel}>Region</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text style={s.rowValue}>{regionLabels[region] ?? region}</Text>
            <Text style={s.rowChevron}>{'\u2192'}</Text>
          </View>
        </Pressable>
        <Pressable
          style={s.row}
          onPress={() => setUnits(units === 'metric' ? 'imperial' : 'metric')}
        >
          <Text style={s.rowLabel}>Units</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text style={s.rowValue}>{units === 'metric' ? 'Metric' : 'Imperial'}</Text>
            <Text style={s.rowChevron}>{'\u2192'}</Text>
          </View>
        </Pressable>
      </View>

      {/* Theme */}
      <View style={s.card}>
        <Text style={s.cardTitle}>Theme</Text>
        <View style={s.themeContainer}>
          {THEME_OPTIONS.map((opt) => {
            const selected = themePreference === opt.value;
            return (
              <Pressable
                key={opt.value}
                style={[
                  s.themePill,
                  selected && { backgroundColor: colors.accent.green },
                ]}
                onPress={() => setThemePreference(opt.value)}
              >
                <Ionicons
                  name={opt.icon}
                  size={16}
                  color={selected ? colors.text.inverse : colors.text.secondary}
                />
                <Text
                  style={[
                    s.themePillLabel,
                    selected && { color: colors.text.inverse },
                  ]}
                >
                  {opt.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>

      {/* Logging Mode */}
      <View style={s.card}>
        <Text style={s.cardTitle}>Logging Mode</Text>
        <Text style={{ fontSize: 13, color: colors.text.tertiary, marginBottom: 12 }}>
          How recipes are logged to your diary
        </Text>
        <UxModeSelector current={uxMode} onSelect={setUxMode} colors={colors} />
      </View>

      {/* Recipes */}
      <View style={s.card}>
        <Text style={s.cardTitle}>Recipes</Text>
        <Pressable style={s.row} onPress={() => rootNavigation.navigate('Recipes')}>
          <Text style={s.rowLabel}>My Recipes</Text>
          <Text style={s.rowChevron}>{'\u2192'}</Text>
        </Pressable>
      </View>

      {/* Export Data */}
      <ExportCard />

      {/* Backups */}
      <BackupCard />

      {/* Google Drive Sync */}
      <SyncCard />

      {/* Gallery Scan */}
      <View style={s.card}>
        <Text style={s.cardTitle}>Gallery Scan</Text>
        <Pressable style={s.row} onPress={() => rootNavigation.navigate('GalleryScan')}>
          <Text style={s.rowLabel}>Gallery Scan Settings</Text>
          <Text style={s.rowChevron}>{'\u2192'}</Text>
        </Pressable>
      </View>

      {/* Notifications */}
      <NotificationsCard />

      {/* Container Weights */}
      <ContainerWeightsCard />

      {/* Health & Weight */}
      <HealthWeightCard />

      {/* AI Models */}
      <View style={s.card}>
        <Text style={s.cardTitle}>AI Models</Text>
        <Pressable style={s.row} onPress={() => rootNavigation.navigate('GeminiNanoTest')}>
          <Text style={s.rowLabel}>Gemini Nano Test</Text>
          <Text style={s.rowChevron}>{'\u2192'}</Text>
        </Pressable>
      </View>

      {/* About */}
      <View style={s.card}>
        <Text style={s.cardTitle}>About</Text>
        <Pressable
          style={s.row}
          onPress={() => Linking.openURL('https://openfoodfacts.org')}
        >
          <View style={{ flex: 1 }}>
            <Text style={s.rowLabel}>Food product data provided by Open Food Facts</Text>
            <Text style={{ fontSize: 12, color: colors.text.tertiary, marginTop: 2 }}>
              Licensed under the Open Database License (ODbL)
            </Text>
          </View>
          <Ionicons name="open-outline" size={16} color={colors.text.tertiary} />
        </Pressable>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function ExportCard() {
  const [exporting, setExporting] = useState(false);
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

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
      const fileUri = `${(FileSystem as any).documentDirectory}${filename}`;

      await (FileSystem as any).writeAsStringAsync(fileUri, content, {
        encoding: (FileSystem as any).EncodingType.UTF8,
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
    <View style={s.card}>
      <Text style={s.cardTitle}>Export Data</Text>
      {exporting ? (
        <ActivityIndicator size="small" color={colors.accent.green} style={{ paddingVertical: 16 }} />
      ) : (
        <>
          <Pressable style={s.exportBtn} onPress={() => handleExport('csv')}>
            <Ionicons name="document-text-outline" size={18} color={colors.accent.green} />
            <Text style={s.exportBtnText}>Export as CSV</Text>
          </Pressable>
          <Pressable style={s.exportBtn} onPress={() => handleExport('json')}>
            <Ionicons name="code-slash-outline" size={18} color={colors.accent.blue} />
            <Text style={[s.exportBtnText, { color: colors.accent.blue }]}>Export as JSON</Text>
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
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

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
    <View style={s.card}>
      <Text style={s.cardTitle}>Backups</Text>

      <View style={s.row}>
        <Text style={s.rowLabel}>Last backup</Text>
        <Text style={s.rowValue}>{lastBackup}</Text>
      </View>
      <View style={s.row}>
        <Text style={s.rowLabel}>Pending changes</Text>
        <Text style={s.rowValue}>{pendingChanges}</Text>
      </View>
      <View style={s.row}>
        <Text style={s.rowLabel}>Total backups</Text>
        <Text style={s.rowValue}>{backups.length}</Text>
      </View>

      {backing ? (
        <ActivityIndicator size="small" color={colors.accent.green} style={{ paddingVertical: 16 }} />
      ) : (
        <>
          <Pressable style={s.exportBtn} onPress={handleIncremental}>
            <Ionicons name="cloud-upload-outline" size={18} color={colors.accent.green} />
            <Text style={s.exportBtnText}>Incremental Backup</Text>
          </Pressable>
          <Pressable style={s.exportBtn} onPress={handleFull}>
            <Ionicons name="download-outline" size={18} color={colors.accent.blue} />
            <Text style={[s.exportBtnText, { color: colors.accent.blue }]}>Full Backup</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

function SyncCard() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { signedIn, userEmail, lastSyncAt, syncStatus } = useSyncStore();
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

  function statusColor(): string {
    switch (syncStatus) {
      case 'syncing': return colors.accent.amber;
      case 'error': return colors.accent.red;
      case 'conflict': return colors.accent.amber;
      default: return colors.accent.green;
    }
  }

  return (
    <Pressable style={s.card} onPress={() => rootNavigation.navigate('SyncSettings')}>
      <View style={s.cardHeader}>
        <Text style={s.cardTitle}>Google Drive Sync</Text>
        <Text style={s.rowChevron}>&#x2192;</Text>
      </View>

      <View style={s.row}>
        <Text style={s.rowLabel}>Account</Text>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <View style={[syncCardStyles.dot, { backgroundColor: signedIn ? colors.accent.green : colors.border.default }]} />
          <Text style={s.rowValue}>{signedIn ? (userEmail ?? 'Connected') : 'Not connected'}</Text>
        </View>
      </View>

      <View style={s.row}>
        <Text style={s.rowLabel}>Last synced</Text>
        <Text style={s.rowValue}>{lastSyncAt ? relativeTime(lastSyncAt) : 'Never'}</Text>
      </View>

      <View style={s.row}>
        <Text style={s.rowLabel}>Status</Text>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          {syncStatus === 'syncing' ? (
            <ActivityIndicator size="small" color={colors.accent.amber} />
          ) : (
            <View style={[syncCardStyles.dot, { backgroundColor: statusColor() }]} />
          )}
          <Text style={s.rowValue}>
            {syncStatus === 'idle' ? 'Up to date' : syncStatus === 'syncing' ? 'Syncing...' : syncStatus === 'error' ? 'Error' : 'Conflicts'}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

const syncCardStyles = StyleSheet.create({
  dot: { width: 8, height: 8, borderRadius: 4 },
});

function NotificationsCard() {
  const {
    notificationsEnabled,
    notificationHour,
    notificationMinute,
    setNotificationsEnabled,
    setNotificationTime,
    nutritionGoals,
  } = usePreferencesStore();
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

  const [hourStr, setHourStr] = useState(String(notificationHour));
  const [minuteStr, setMinuteStr] = useState(String(notificationMinute).padStart(2, '0'));

  async function handleToggle(enabled: boolean) {
    if (enabled) {
      const granted = await requestNotificationPermission();
      if (!granted) {
        Alert.alert('Permission Denied', 'Notification permission is required.');
        return;
      }
      const body = buildMacroSummaryBody(nutritionGoals);
      await scheduleDailyNotification(notificationHour, notificationMinute, body);
    } else {
      await cancelDailyNotification();
    }
    setNotificationsEnabled(enabled);
  }

  async function handleTimeChange() {
    const h = parseInt(hourStr, 10);
    const m = parseInt(minuteStr, 10);
    if (isNaN(h) || h < 0 || h > 23 || isNaN(m) || m < 0 || m > 59) {
      Alert.alert('Invalid Time', 'Hour must be 0-23, minute 0-59.');
      return;
    }
    setNotificationTime(h, m);
    if (notificationsEnabled) {
      const body = buildMacroSummaryBody(nutritionGoals);
      await scheduleDailyNotification(h, m, body);
    }
  }

  return (
    <View style={s.card}>
      <Text style={s.cardTitle}>Notifications</Text>
      <View style={s.row}>
        <Text style={s.rowLabel}>Daily Summary</Text>
        <Switch
          value={notificationsEnabled}
          onValueChange={handleToggle}
          trackColor={{ true: colors.accent.green, false: colors.border.default }}
          thumbColor={colors.background.elevated}
        />
      </View>
      {notificationsEnabled && (
        <View style={[s.row, { gap: 8 }]}>
          <Text style={s.rowLabel}>Time</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}>
            <TextInput
              style={{
                backgroundColor: colors.input.background,
                borderRadius: 8,
                paddingHorizontal: 10,
                paddingVertical: 6,
                fontSize: 16,
                fontWeight: '600',
                color: colors.text.primary,
                width: 44,
                textAlign: 'center',
              }}
              value={hourStr}
              onChangeText={setHourStr}
              onBlur={handleTimeChange}
              keyboardType="number-pad"
              maxLength={2}
            />
            <Text style={{ fontSize: 16, fontWeight: '700', color: colors.text.secondary }}>:</Text>
            <TextInput
              style={{
                backgroundColor: colors.input.background,
                borderRadius: 8,
                paddingHorizontal: 10,
                paddingVertical: 6,
                fontSize: 16,
                fontWeight: '600',
                color: colors.text.primary,
                width: 44,
                textAlign: 'center',
              }}
              value={minuteStr}
              onChangeText={setMinuteStr}
              onBlur={handleTimeChange}
              keyboardType="number-pad"
              maxLength={2}
            />
          </View>
        </View>
      )}
    </View>
  );
}

function ContainerWeightsCard() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [containers, setContainers] = useState<Container[]>([]);
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);

  useEffect(() => {
    loadContainers();
  }, []);

  async function loadContainers() {
    try {
      const list = await getContainers();
      setContainers(list);
    } catch {
      // Non-critical
    }
  }

  async function handleDelete(id: number, name: string) {
    Alert.alert('Delete Container', `Delete "${name}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await deleteContainer(id);
            await loadContainers();
          } catch {
            Alert.alert('Error', 'Failed to delete container.');
          }
        },
      },
    ]);
  }

  return (
    <View style={s.card}>
      <View style={s.cardHeader}>
        <Text style={s.cardTitle}>Container Weights</Text>
        <Pressable onPress={() => rootNavigation.navigate('ScaleInput', {})}>
          <Text style={s.editBtn}>Manage</Text>
        </Pressable>
      </View>
      {containers.length === 0 ? (
        <Text style={{ fontSize: 13, color: colors.text.tertiary, paddingVertical: 8 }}>
          No containers saved. Add containers from the Scale Input screen.
        </Text>
      ) : (
        containers.map((c) => (
          <Pressable
            key={c.id}
            style={s.row}
            onLongPress={() => handleDelete(c.id, c.name)}
          >
            <Text style={s.rowLabel}>{c.name}</Text>
            <Text style={s.rowValue}>
              {c.weightGrams}g {c.timesUsed > 0 ? `(used ${c.timesUsed}x)` : ''}
            </Text>
          </Pressable>
        ))
      )}
    </View>
  );
}

function HealthWeightCard() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const {
    healthConnectEnabled,
    setHealthConnectEnabled,
  } = usePreferencesStore();
  const { colors } = useTheme();
  const s = useMemo(() => createStyles(colors), [colors]);
  const [hcAvailable, setHcAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    checkAvailability();
  }, []);

  async function checkAvailability() {
    const available = await isHealthConnectAvailable();
    setHcAvailable(available);
  }

  async function handleToggle(enabled: boolean) {
    if (enabled) {
      try {
        await initHealthConnect();
      } catch {
        Alert.alert('Error', 'Failed to initialize Health Connect. Make sure it is installed and up to date.');
        return;
      }
      try {
        const granted = await requestWeightPermission();
        if (!granted) {
          Alert.alert('Permission Denied', 'Health Connect weight read permission is required.');
          return;
        }
      } catch {
        Alert.alert('Error', 'Failed to request Health Connect permissions.');
        return;
      }
    }
    setHealthConnectEnabled(enabled);
  }

  return (
    <View style={s.card}>
      <Text style={s.cardTitle}>Health & Weight</Text>

      <View style={s.row}>
        <Text style={s.rowLabel}>Health Connect</Text>
        {hcAvailable === false ? (
          <Text style={{ fontSize: 13, color: colors.accent.red }}>Not available</Text>
        ) : (
          <Switch
            value={healthConnectEnabled}
            onValueChange={handleToggle}
            trackColor={{ true: colors.accent.green, false: colors.border.default }}
            thumbColor={colors.background.elevated}
          />
        )}
      </View>

      {hcAvailable === false && (
        <Text style={{ fontSize: 12, color: colors.text.tertiary, paddingBottom: 8 }}>
          Install Google Health Connect from the Play Store (required for Android &lt; 14).
        </Text>
      )}

      <Pressable
        style={s.row}
        onPress={() => rootNavigation.navigate('WeightTrend')}
      >
        <Text style={s.rowLabel}>View Weight Trend</Text>
        <Text style={s.rowChevron}>{'\u2192'}</Text>
      </Pressable>
    </View>
  );
}

const UX_MODE_OPTIONS: { mode: UxMode; label: string; desc: string }[] = [
  { mode: 'zero-effort', label: 'Zero-effort', desc: 'Auto-log, review later' },
  { mode: 'confirm-only', label: 'Confirm', desc: 'Review before logging' },
  { mode: 'guided-edit', label: 'Guided', desc: 'Step-by-step editing' },
];

function UxModeSelector({ current, onSelect, colors }: { current: UxMode; onSelect: (m: UxMode) => void; colors: ThemeColors }) {
  const s = useMemo(() => createStyles(colors), [colors]);
  return (
    <View style={s.uxModeContainer}>
      {UX_MODE_OPTIONS.map(({ mode, label, desc }) => {
        const selected = current === mode;
        return (
          <Pressable
            key={mode}
            style={[s.uxModeOption, selected && s.uxModeOptionSelected]}
            onPress={() => onSelect(mode)}
          >
            <View style={[s.uxModeRadio, selected && s.uxModeRadioSelected]}>
              {selected && <View style={s.uxModeRadioDot} />}
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[s.uxModeLabel, selected && s.uxModeLabelSelected]}>{label}</Text>
              <Text style={s.uxModeDesc}>{desc}</Text>
            </View>
          </Pressable>
        );
      })}
    </View>
  );
}

function GoalRow({ label, unit, value, editing, onChange, color, colors }: {
  label: string; unit: string; value: string; editing: boolean;
  onChange: (v: string) => void; color: string; colors: ThemeColors;
}) {
  const s = useMemo(() => createStyles(colors), [colors]);
  return (
    <View style={s.goalRow}>
      <View style={s.goalLabelRow}>
        <View style={[s.goalDot, { backgroundColor: color }]} />
        <Text style={s.goalLabel}>{label}</Text>
      </View>
      {editing ? (
        <View style={s.goalInputRow}>
          <TextInput
            style={s.goalInput}
            value={value}
            onChangeText={onChange}
            keyboardType="number-pad"
            selectTextOnFocus
          />
          <Text style={s.goalUnit}>{unit}</Text>
        </View>
      ) : (
        <Text style={s.goalValue}>{value} {unit}</Text>
      )}
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.background.primary },
    content: { paddingTop: 60, paddingHorizontal: 16 },
    title: { fontSize: 28, fontWeight: '800', color: colors.text.primary, marginBottom: 20 },

    card: {
      backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16, marginBottom: 16,
      shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
      shadowRadius: 8, elevation: 3,
    },
    cardHeader: {
      flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12,
    },
    cardTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
    editBtn: { fontSize: 15, fontWeight: '600', color: colors.accent.blue },
    editActions: { flexDirection: 'row', gap: 16 },
    cancelBtn: { fontSize: 15, fontWeight: '500', color: colors.text.tertiary },
    saveBtn: { fontSize: 15, fontWeight: '700', color: colors.accent.green },

    goalRow: {
      flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
      paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.background.surface,
    },
    goalLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    goalDot: { width: 10, height: 10, borderRadius: 5 },
    goalLabel: { fontSize: 15, fontWeight: '500', color: colors.text.secondary },
    goalValue: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
    goalInputRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    goalInput: {
      backgroundColor: colors.input.background, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
      fontSize: 15, fontWeight: '600', color: colors.text.primary, minWidth: 70, textAlign: 'right',
    },
    goalUnit: { fontSize: 13, color: colors.text.tertiary },

    row: {
      flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
      paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.background.surface,
    },
    rowLabel: { fontSize: 15, color: colors.text.secondary },
    rowValue: { fontSize: 15, color: colors.text.tertiary },
    rowChevron: { fontSize: 16, color: colors.border.default },
    exportBtn: {
      flexDirection: 'row', alignItems: 'center', gap: 8,
      paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.background.surface,
    },
    exportBtnText: { fontSize: 15, fontWeight: '500', color: colors.accent.green },

    // Theme selector
    themeContainer: { flexDirection: 'row', gap: 8, marginTop: 8 },
    themePill: {
      flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
      paddingVertical: 10, borderRadius: 12,
      backgroundColor: colors.background.surface,
    },
    themePillLabel: { fontSize: 14, fontWeight: '600', color: colors.text.secondary },

    // UX Mode selector
    uxModeContainer: { gap: 8 },
    uxModeOption: {
      flexDirection: 'row', alignItems: 'center', gap: 12,
      paddingVertical: 12, paddingHorizontal: 12, borderRadius: 12,
      borderWidth: 1, borderColor: colors.border.subtle, backgroundColor: colors.background.surface,
    },
    uxModeOptionSelected: {
      borderColor: colors.accent.purple, backgroundColor: colors.accentTint.purple,
    },
    uxModeRadio: {
      width: 20, height: 20, borderRadius: 10,
      borderWidth: 2, borderColor: colors.border.default,
      justifyContent: 'center', alignItems: 'center',
    },
    uxModeRadioSelected: { borderColor: colors.accent.purple },
    uxModeRadioDot: {
      width: 10, height: 10, borderRadius: 5, backgroundColor: colors.accent.purple,
    },
    uxModeLabel: { fontSize: 15, fontWeight: '600', color: colors.text.secondary },
    uxModeLabelSelected: { color: colors.accent.purple },
    uxModeDesc: { fontSize: 12, color: colors.text.tertiary, marginTop: 1 },
  });
}
