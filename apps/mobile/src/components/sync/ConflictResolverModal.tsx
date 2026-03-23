/**
 * ConflictResolverModal -- per-field conflict resolution UI.
 *
 * Iterates pendingConflicts from useSyncStore, shows local vs remote values,
 * and lets user pick per-field or bulk "Keep Latest" resolution.
 */

import React, { useMemo, useState } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  Alert,
} from 'react-native';
import { useSyncStore } from '../../store/useSyncStore';
import { applyResolution, autoResolveConflicts } from '../../services/sync/conflictResolver';
import type { SyncConflict, SyncResolution } from '../../services/sync/types';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface Props {
  visible: boolean;
  onClose: () => void;
}

export default function ConflictResolverModal({ visible, onClose }: Props) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const { pendingConflicts, setPendingConflicts } = useSyncStore();
  const [resolving, setResolving] = useState(false);

  function resolveConflict(conflict: SyncConflict, source: 'local' | 'remote') {
    const resolution: SyncResolution = {
      table: conflict.table,
      rowId: conflict.rowId,
      field: conflict.field,
      resolvedValue: source === 'local' ? conflict.localValue : conflict.remoteValue,
      source,
    };

    try {
      applyResolution([resolution]);
    } catch {
      Alert.alert('Error', 'Failed to apply resolution.');
      return;
    }

    const remaining = pendingConflicts.filter(
      (c) => !(c.table === conflict.table && c.rowId === conflict.rowId && c.field === conflict.field),
    );
    setPendingConflicts(remaining);

    if (remaining.length === 0) {
      Alert.alert('All Resolved', 'All conflicts have been resolved.');
      onClose();
    }
  }

  function resolveAllKeepLatest() {
    setResolving(true);

    try {
      const resolutions = autoResolveConflicts(pendingConflicts);
      applyResolution(resolutions);
    } catch {
      setResolving(false);
      Alert.alert('Error', 'Failed to apply resolutions.');
      return;
    }

    setPendingConflicts([]);
    setResolving(false);
    Alert.alert('All Resolved', 'All conflicts resolved using most recent values.');
    onClose();
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Resolve Conflicts</Text>
          <Pressable onPress={onClose}>
            <Text style={styles.closeBtn}>Close</Text>
          </Pressable>
        </View>

        {pendingConflicts.length > 0 && (
          <Pressable style={styles.bulkBtn} onPress={resolveAllKeepLatest}>
            <Text style={styles.bulkBtnText}>Resolve All -- Keep Latest</Text>
          </Pressable>
        )}

        <ScrollView style={styles.list} contentContainerStyle={{ paddingBottom: 40 }}>
          {pendingConflicts.length === 0 ? (
            <Text style={styles.emptyText}>No conflicts to resolve.</Text>
          ) : (
            pendingConflicts.map((conflict, idx) => (
              <ConflictCard
                key={`${conflict.table}-${conflict.rowId}-${conflict.field}-${idx}`}
                conflict={conflict}
                onResolve={resolveConflict}
                colors={colors}
              />
            ))
          )}
        </ScrollView>
      </View>
    </Modal>
  );
}

function ConflictCard({
  conflict,
  onResolve,
  colors,
}: {
  conflict: SyncConflict;
  onResolve: (c: SyncConflict, source: 'local' | 'remote') => void;
  colors: ThemeColors;
}) {
  return (
    <View style={cardStyles(colors).card}>
      <Text style={cardStyles(colors).cardTable}>
        {conflict.table} (row {conflict.rowId})
      </Text>
      <Text style={cardStyles(colors).cardField}>Field: {conflict.field}</Text>

      <View style={cardStyles(colors).valuesRow}>
        <View style={cardStyles(colors).valueBox}>
          <Text style={cardStyles(colors).valueLabel}>Local</Text>
          <Text style={cardStyles(colors).valueText} numberOfLines={3}>
            {String(conflict.localValue)}
          </Text>
          <Text style={cardStyles(colors).timestamp}>
            {new Date(conflict.localTimestamp).toLocaleString()}
          </Text>
          <Pressable
            style={[cardStyles(colors).resolveBtn, { backgroundColor: colors.accentTint.blue }]}
            onPress={() => onResolve(conflict, 'local')}
          >
            <Text style={cardStyles(colors).resolveBtnText}>Keep Local</Text>
          </Pressable>
        </View>

        <View style={cardStyles(colors).valueBox}>
          <Text style={cardStyles(colors).valueLabel}>Remote</Text>
          <Text style={cardStyles(colors).valueText} numberOfLines={3}>
            {String(conflict.remoteValue)}
          </Text>
          <Text style={cardStyles(colors).timestamp}>
            {new Date(conflict.remoteTimestamp).toLocaleString()}
          </Text>
          <Pressable
            style={[cardStyles(colors).resolveBtn, { backgroundColor: colors.accentTint.red }]}
            onPress={() => onResolve(conflict, 'remote')}
          >
            <Text style={cardStyles(colors).resolveBtnText}>Keep Remote</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const cardStyles = (colors: ThemeColors) => StyleSheet.create({
  card: {
    backgroundColor: colors.background.elevated,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTable: { fontSize: 14, fontWeight: '700', color: colors.text.secondary, marginBottom: 4 },
  cardField: { fontSize: 13, color: colors.text.tertiary, marginBottom: 12 },
  valuesRow: { flexDirection: 'row', gap: 12 },
  valueBox: { flex: 1 },
  valueLabel: { fontSize: 12, fontWeight: '600', color: colors.text.tertiary, marginBottom: 4 },
  valueText: { fontSize: 14, color: colors.text.primary, marginBottom: 4 },
  timestamp: { fontSize: 11, color: colors.text.tertiary, marginBottom: 8 },
  resolveBtn: {
    borderRadius: 8,
    paddingVertical: 8,
    alignItems: 'center',
  },
  resolveBtnText: { fontSize: 13, fontWeight: '600', color: colors.text.secondary },
});

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.background.primary },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingTop: 60,
      paddingHorizontal: 16,
      paddingBottom: 12,
      backgroundColor: colors.background.elevated,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border.subtle,
    },
    title: { fontSize: 20, fontWeight: '700', color: colors.text.primary },
    closeBtn: { fontSize: 16, fontWeight: '600', color: colors.accent.blue },

    bulkBtn: {
      marginHorizontal: 16,
      marginTop: 12,
      backgroundColor: colors.accent.purple,
      borderRadius: 12,
      paddingVertical: 14,
      alignItems: 'center',
    },
    bulkBtnText: { fontSize: 15, fontWeight: '700', color: colors.text.inverse },

    list: { flex: 1, paddingHorizontal: 16, marginTop: 12 },
    emptyText: { fontSize: 15, color: colors.text.tertiary, textAlign: 'center', marginTop: 40 },
  });
}
