/**
 * WeightTrendScreen -- Weight tracking with EMA trend display.
 *
 * Shows raw weight entries, smoothed EMA values, trend direction,
 * and manual weight entry form. Optional Health Connect sync.
 */

import React, {useEffect, useState, useMemo} from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { useWeightStore } from '../store/useWeightStore';
import { usePreferencesStore } from '../store/usePreferencesStore';
import {
  isHealthConnectAvailable,
  initHealthConnect,
  requestWeightPermission,
} from '../services/health/healthConnectService';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

const TREND_ICONS: Record<string, string> = {
  up: '\u2191',    // arrow up
  down: '\u2193',  // arrow down
  stable: '\u2192', // arrow right
};

function getTrendColors(colors: ThemeColors): Record<string, string> {
  return {
    up: colors.accent.red,
    down: colors.accent.green,
    stable: colors.text.tertiary,
  };
}

export default function WeightTrendScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const TREND_COLORS = useMemo(() => getTrendColors(colors), [colors]);

  const navigation = useNavigation();
  const { entries, isLoading, loadEntries, addManualWeight, syncFromHealthConnect } =
    useWeightStore();
  const trend = useWeightStore((s) => s.getWeightTrend());
  const { healthConnectEnabled } = usePreferencesStore();

  // Manual entry form
  const [weightInput, setWeightInput] = useState('');
  const [dateInput, setDateInput] = useState(
    new Date().toISOString().split('T')[0],
  );
  const [syncing, setSyncing] = useState(false);
  const [hcAvailable, setHcAvailable] = useState(false);

  useEffect(() => {
    loadEntries();
    checkHealthConnect();
  }, []);

  async function checkHealthConnect() {
    const available = await isHealthConnectAvailable();
    setHcAvailable(available);
  }

  async function handleAddWeight() {
    const kg = parseFloat(weightInput);
    if (isNaN(kg) || kg <= 0 || kg > 500) {
      Alert.alert('Invalid Weight', 'Enter a weight between 0 and 500 kg.');
      return;
    }
    // Validate date format YYYY-MM-DD
    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateInput)) {
      Alert.alert('Invalid Date', 'Enter a date in YYYY-MM-DD format.');
      return;
    }
    await addManualWeight(dateInput, kg);
    setWeightInput('');
    Alert.alert('Added', `${kg} kg on ${dateInput}`);
  }

  async function handleSync() {
    setSyncing(true);
    try {
      await initHealthConnect();
      const granted = await requestWeightPermission();
      if (!granted) {
        Alert.alert('Permission Denied', 'Health Connect weight read permission is required.');
        return;
      }
      const count = await syncFromHealthConnect();
      Alert.alert('Synced', `Imported ${count} weight records from Health Connect.`);
    } catch (err) {
      Alert.alert('Sync Failed', err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setSyncing(false);
    }
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => navigation.goBack()} style={styles.headerBack}>
          <Text style={styles.headerBackText}>{'\u2190'}</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Weight Trend</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* Trend Summary */}
        {trend.latestKg !== null ? (
          <View style={styles.summaryCard}>
            <View style={styles.summaryRow}>
              <View>
                <Text style={styles.summaryLabel}>Latest</Text>
                <Text style={styles.summaryWeight}>
                  {trend.latestKg.toFixed(1)} kg
                </Text>
              </View>
              <View style={styles.trendBadge}>
                <Text
                  style={[
                    styles.trendArrow,
                    { color: TREND_COLORS[trend.trendDirection] },
                  ]}
                >
                  {TREND_ICONS[trend.trendDirection]}
                </Text>
                <Text
                  style={[
                    styles.trendLabel,
                    { color: TREND_COLORS[trend.trendDirection] },
                  ]}
                >
                  {trend.trendDirection === 'up'
                    ? 'Trending up'
                    : trend.trendDirection === 'down'
                      ? 'Trending down'
                      : 'Stable'}
                </Text>
              </View>
            </View>
            {trend.smoothed.length > 0 && (
              <Text style={styles.summarySmoothed}>
                Smoothed: {trend.smoothed[trend.smoothed.length - 1].toFixed(1)} kg
              </Text>
            )}
          </View>
        ) : (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyIcon}>{'  \u2696\uFE0F'}</Text>
            <Text style={styles.emptyTitle}>No weight data yet</Text>
            <Text style={styles.emptySubtitle}>
              Add your first weight entry below to start tracking your trend.
            </Text>
          </View>
        )}

        {/* Simple chart: dots-based trend visualization */}
        {trend.raw.length >= 2 && (
          <View style={styles.chartCard}>
            <Text style={styles.sectionTitle}>Trend Chart</Text>
            <SimpleChart
              raw={trend.raw}
              smoothed={trend.smoothed}
              dates={trend.dates}
            />
          </View>
        )}

        {/* Manual Entry */}
        <View style={styles.entryCard}>
          <Text style={styles.sectionTitle}>Add Weight</Text>
          <View style={styles.entryRow}>
            <TextInput
              style={styles.dateInput}
              value={dateInput}
              onChangeText={setDateInput}
              placeholder="YYYY-MM-DD"
              placeholderTextColor="#9CA3AF"
            />
            <TextInput
              style={styles.weightInputField}
              value={weightInput}
              onChangeText={setWeightInput}
              keyboardType="decimal-pad"
              placeholder="kg"
              placeholderTextColor="#9CA3AF"
            />
            <Pressable style={styles.addBtn} onPress={handleAddWeight}>
              <Text style={styles.addBtnText}>Add</Text>
            </Pressable>
          </View>
        </View>

        {/* Health Connect Sync */}
        {healthConnectEnabled && hcAvailable && (
          <View style={styles.syncCard}>
            <Pressable
              style={styles.syncBtn}
              onPress={handleSync}
              disabled={syncing}
            >
              {syncing ? (
                <ActivityIndicator size="small" color={colors.text.inverse} />
              ) : (
                <Text style={styles.syncBtnText}>Sync from Health Connect</Text>
              )}
            </Pressable>
          </View>
        )}

        {/* History List */}
        {entries.length > 0 && (
          <View style={styles.historyCard}>
            <Text style={styles.sectionTitle}>History</Text>
            {isLoading ? (
              <ActivityIndicator size="small" color={colors.accent.green} />
            ) : (
              [...entries].reverse().map((entry, idx) => {
                const smoothedIdx = entries.length - 1 - idx;
                const smoothedVal = trend.smoothed[smoothedIdx];
                return (
                  <View key={`${entry.date}-${idx}`} style={styles.historyRow}>
                    <Text style={styles.historyDate}>{entry.date}</Text>
                    <View style={styles.historyValues}>
                      <Text style={styles.historyRaw}>
                        {entry.weightKg.toFixed(1)} kg
                      </Text>
                      {smoothedVal != null && (
                        <Text style={styles.historySmoothed}>
                          EMA: {smoothedVal.toFixed(1)}
                        </Text>
                      )}
                    </View>
                    <Text style={styles.historySource}>
                      {entry.source === 'health_connect' ? 'HC' : 'Manual'}
                    </Text>
                  </View>
                );
              })
            )}
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

/**
 * Minimal View-based chart. Plots raw dots and smoothed line approximation
 * using absolute-positioned Views.
 */
function SimpleChart({
  raw,
  smoothed,
  dates,
}: {
  raw: number[];
  smoothed: number[];
  dates: string[];
}) {
  const { colors } = useTheme();
  const chartStyles = useMemo(() => createChartStyles(colors), [colors]);
  const CHART_HEIGHT = 160;
  const CHART_PADDING = 16;

  if (raw.length < 2) return null;

  const minVal = Math.min(...raw, ...smoothed) - 0.5;
  const maxVal = Math.max(...raw, ...smoothed) + 0.5;
  const range = maxVal - minVal || 1;

  function yPos(val: number): number {
    return CHART_HEIGHT - ((val - minVal) / range) * (CHART_HEIGHT - CHART_PADDING * 2) - CHART_PADDING;
  }

  const pointWidth = Math.max(4, Math.min(20, (300 - 32) / raw.length));

  return (
    <View style={[chartStyles.container, { height: CHART_HEIGHT }]}>
      {/* Y-axis labels */}
      <Text style={[chartStyles.yLabel, { top: CHART_PADDING - 6 }]}>
        {maxVal.toFixed(0)}
      </Text>
      <Text style={[chartStyles.yLabel, { bottom: CHART_PADDING - 6 }]}>
        {minVal.toFixed(0)}
      </Text>

      {/* Raw data points (blue dots) */}
      {raw.map((val, i) => (
        <View
          key={`raw-${i}`}
          style={[
            chartStyles.dot,
            {
              left: 30 + i * pointWidth,
              top: yPos(val) - 3,
              backgroundColor: '#93C5FD',
            },
          ]}
        />
      ))}

      {/* Smoothed data points (green dots, larger) */}
      {smoothed.map((val, i) => (
        <View
          key={`smooth-${i}`}
          style={[
            chartStyles.smoothDot,
            {
              left: 30 + i * pointWidth - 1,
              top: yPos(val) - 4,
              backgroundColor: colors.accent.green,
            },
          ]}
        />
      ))}

      {/* Legend */}
      <View style={chartStyles.legend}>
        <View style={chartStyles.legendItem}>
          <View style={[chartStyles.legendDot, { backgroundColor: '#93C5FD' }]} />
          <Text style={chartStyles.legendText}>Raw</Text>
        </View>
        <View style={chartStyles.legendItem}>
          <View style={[chartStyles.legendDot, { backgroundColor: colors.accent.green }]} />
          <Text style={chartStyles.legendText}>EMA</Text>
        </View>
      </View>
    </View>
  );
}

function createChartStyles(colors: ThemeColors) {
  return StyleSheet.create({
  container: {
    position: 'relative',
    marginTop: 8,
    backgroundColor: colors.background.surface,
    borderRadius: 12,
    overflow: 'hidden',
  },
  yLabel: {
    position: 'absolute',
    left: 4,
    fontSize: 10,
    color: colors.text.tertiary,
  },
  dot: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  smoothDot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legend: {
    position: 'absolute',
    top: 4,
    right: 8,
    flexDirection: 'row',
    gap: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    fontSize: 10,
    color: colors.text.tertiary,
  },
});
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background.primary },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: colors.background.elevated, paddingHorizontal: 16, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
  },
  headerBack: { padding: 4, width: 36 },
  headerBackText: { fontSize: 22, color: colors.text.tertiary, fontWeight: '600' },
  headerTitle: { fontSize: 17, fontWeight: '700', color: colors.text.primary },
  headerRight: { width: 36 },
  scrollView: { flex: 1 },
  scrollContent: { padding: 16 },

  summaryCard: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 20, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  summaryRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  summaryLabel: { fontSize: 13, color: colors.text.tertiary, marginBottom: 4 },
  summaryWeight: { fontSize: 32, fontWeight: '800', color: colors.text.primary },
  summarySmoothed: { fontSize: 13, color: colors.text.tertiary, marginTop: 8 },
  trendBadge: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  trendArrow: { fontSize: 24, fontWeight: '800' },
  trendLabel: { fontSize: 14, fontWeight: '600' },

  emptyCard: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 32, marginBottom: 16,
    alignItems: 'center',
  },
  emptyIcon: { fontSize: 40, marginBottom: 12 },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary, marginBottom: 8 },
  emptySubtitle: { fontSize: 14, color: colors.text.tertiary, textAlign: 'center', lineHeight: 20 },

  chartCard: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },

  sectionTitle: { fontSize: 15, fontWeight: '600', color: colors.text.secondary, marginBottom: 8 },

  entryCard: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  entryRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dateInput: {
    flex: 1, backgroundColor: colors.background.surface, borderRadius: 8,
    paddingHorizontal: 12, paddingVertical: 10, fontSize: 14, color: colors.text.primary,
  },
  weightInputField: {
    width: 80, backgroundColor: colors.background.surface, borderRadius: 8,
    paddingHorizontal: 12, paddingVertical: 10, fontSize: 14, color: colors.text.primary,
    textAlign: 'center',
  },
  addBtn: {
    backgroundColor: colors.accent.green, borderRadius: 8, paddingHorizontal: 16, paddingVertical: 10,
  },
  addBtnText: { fontSize: 14, fontWeight: '600', color: colors.text.inverse },

  syncCard: {
    marginBottom: 16,
  },
  syncBtn: {
    backgroundColor: colors.accent.blue, borderRadius: 14, paddingVertical: 14, alignItems: 'center',
  },
  syncBtnText: { color: colors.text.inverse, fontSize: 15, fontWeight: '600' },

  historyCard: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  historyRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingVertical: 10, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.background.surface,
  },
  historyDate: { fontSize: 14, color: colors.text.secondary, fontWeight: '500', width: 100 },
  historyValues: { flex: 1, alignItems: 'flex-end' },
  historyRaw: { fontSize: 15, fontWeight: '700', color: colors.text.primary },
  historySmoothed: { fontSize: 12, color: colors.text.tertiary, marginTop: 2 },
  historySource: { fontSize: 11, color: colors.text.tertiary, width: 50, textAlign: 'right' },
});
}
